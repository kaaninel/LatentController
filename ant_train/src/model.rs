/// ANT transformer model — pure neural network, no trie.
///
/// Components (same as Python model.py):
///   RMSNorm, RoPE, SiLUFFN, AddrNet (8-cycle co-processor),
///   Attention (causal, KV-cache), MemoryAttention (cross-attn),
///   TransformerBlock (self+tag+mem+ffn), ANT top-level.
///
/// LM head is weight-tied with byte embedding.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{embedding, linear_no_bias, ops, Embedding, Linear, Module, VarBuilder};
use rand::Rng;

use crate::config::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn causal_mask(seq_len: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len * seq_len)
        .map(|i| if (i % seq_len) <= (i / seq_len) { 0.0 } else { f32::NEG_INFINITY })
        .collect();
    Tensor::from_vec(mask, (seq_len, seq_len), device)
}

fn sdpa(q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
    let scale = (q.dim(D::Minus1)? as f64).sqrt();
    // q: (B, H, T, D), k: (B, H, S, D)
    let kt = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
    let attn = q.matmul(&kt)?;
    let attn = (attn / scale)?;
    let attn = match mask {
        Some(m) => attn.broadcast_add(m)?,
        None => attn,
    };
    let attn = ops::softmax(&attn, D::Minus1)?;
    attn.matmul(&v.contiguous()?)
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn new(d_model: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(d_model, "weight")?;
        Ok(Self { weight, eps: 1e-6 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x_f = x.to_dtype(DType::F32)?;
        let rms = x_f.sqr()?.mean_keepdim(D::Minus1)?.affine(1.0, self.eps)?.sqrt()?.recip()?;
        let normed = x_f.broadcast_mul(&rms)?;
        normed.broadcast_mul(&self.weight.to_dtype(DType::F32)?)?.to_dtype(dtype)
    }
}

// ---------------------------------------------------------------------------
// RoPE
// ---------------------------------------------------------------------------

pub struct RoPE {
    cos: Tensor,
    sin: Tensor,
}

impl RoPE {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32, device: &Device) -> Result<Self> {
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / theta.powf(i as f32 / half as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (half,), device)?;
        let t: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let t = Tensor::from_vec(t, (max_seq_len,), device)?;
        // freqs: (max_seq_len, half)
        let freqs = t.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
        Ok(Self { cos: freqs.cos()?, sin: freqs.sin()? })
    }

    pub fn apply(&self, x: &Tensor, offset: usize, seq_len: usize) -> Result<Tensor> {
        let half = x.dim(D::Minus1)? / 2;
        let cos = self.cos.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, offset, seq_len)?.unsqueeze(0)?.unsqueeze(0)?;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
        Tensor::cat(&[&r1, &r2], D::Minus1)
    }
}

// ---------------------------------------------------------------------------
// SiLU FFN
// ---------------------------------------------------------------------------

pub struct SiLUFFN {
    up: Linear,
    down: Linear,
}

impl SiLUFFN {
    pub fn new(d_model: usize, ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            up: linear_no_bias(d_model, ffn_dim, vb.pp("up"))?,
            down: linear_no_bias(ffn_dim, d_model, vb.pp("down"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.up.forward(x)?;
        // SiLU: x * sigmoid(x)
        let sig = ops::sigmoid(&h)?;
        let h = h.mul(&sig)?;
        self.down.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// AddrNet — 8-cycle hierarchical address co-processor
// ---------------------------------------------------------------------------

pub struct AddrNet {
    proj_in: Linear,
    bin_embed: Embedding,
    mlp: Linear,
    out: Linear,
}

impl AddrNet {
    pub fn new(d_model: usize, hidden_dim: usize, n_bins: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            proj_in: linear_no_bias(d_model, hidden_dim, vb.pp("proj_in"))?,
            bin_embed: embedding(n_bins, hidden_dim, vb.pp("bin_embed"))?,
            mlp: linear_no_bias(hidden_dim, hidden_dim, vb.pp("mlp"))?,
            out: linear_no_bias(hidden_dim, n_bins, vb.pp("out"))?,
        })
    }

    /// hidden: (B, d_model) → addresses: (B, depth) as u32 tensor
    /// training=true: soft Gumbel-softmax for differentiable addr selection
    /// training=false: argmax (deterministic)
    pub fn forward(&self, hidden: &Tensor, temperature: f32, training: bool, rng: &mut impl Rng)
        -> Result<(Tensor, Vec<Tensor>)>
    {
        let mut h = self.proj_in.forward(hidden)?; // (B, hidden_dim)
        let mut bins: Vec<Tensor> = Vec::with_capacity(ADDR_DEPTH);
        let mut all_logits: Vec<Tensor> = Vec::with_capacity(ADDR_DEPTH);

        for _ in 0..ADDR_DEPTH {
            let logits = self.out.forward(&h)?; // (B, n_bins)
            // Clamp logits
            let logits = logits.clamp(-30.0f32, 30.0f32)?;
            all_logits.push(logits.clone());

            let (bin_idx, embed_input) = if training && temperature > 0.0 {
                // Gumbel-softmax: add Gumbel noise, take softmax, use as soft one-hot
                let shape = logits.shape().clone();
                let u_vals: Vec<f32> = (0..shape.elem_count())
                    .map(|_| {
                        let u: f32 = rng.gen::<f32>().max(1e-20).min(1.0 - 1e-20);
                        -(-u.ln()).ln()
                    })
                    .collect();
                let gumbel = Tensor::from_vec(u_vals, shape, logits.device())?;
                let noisy = ((logits + gumbel)? / temperature as f64)?;
                let idx = noisy.argmax(D::Minus1)?; // (B,) u32
                // Soft weights for differentiable embedding lookup
                let soft = ops::softmax(&noisy, D::Minus1)?; // (B, n_bins)
                // Embed: soft @ bin_embed.weight => (B, hidden_dim)
                let embed_w = self.bin_embed.embeddings(); // (n_bins, hidden_dim)
                let soft_embed = soft.matmul(embed_w)?;
                (idx, soft_embed)
            } else {
                let idx = logits.argmax(D::Minus1)?;
                let hard_embed = self.bin_embed.forward(&idx)?;
                (idx, hard_embed)
            };

            h = (h + embed_input)?;
            // SiLU(mlp(h))
            let h_mlp = self.mlp.forward(&h)?;
            let sig = ops::sigmoid(&h_mlp)?;
            h = h_mlp.mul(&sig)?;

            bins.push(bin_idx);
        }

        // Stack: (B, depth) — u32
        let addrs = Tensor::stack(&bins, 1)?;
        Ok((addrs, all_logits))
    }
}

// ---------------------------------------------------------------------------
// Attention (causal self-attention with KV cache)
// ---------------------------------------------------------------------------

pub struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl Attention {
    pub fn new(d_model: usize, n_heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner = n_heads * head_dim;
        Ok(Self {
            q: linear_no_bias(d_model, inner, vb.pp("q"))?,
            k: linear_no_bias(d_model, inner, vb.pp("k"))?,
            v: linear_no_bias(d_model, inner, vb.pp("v"))?,
            o: linear_no_bias(inner, d_model, vb.pp("o"))?,
            n_heads,
            head_dim,
        })
    }

    /// x: (B, T, d), returns (out, k_cache, v_cache)
    pub fn forward(&self, x: &Tensor, rope: &RoPE, rope_offset: usize,
                   causal: bool, k_cache: Option<&Tensor>, v_cache: Option<&Tensor>)
        -> Result<(Tensor, Tensor, Tensor)>
    {
        let (b, t, _) = x.dims3()?;
        let h = self.n_heads;
        let d = self.head_dim;

        let q = self.q.forward(x)?.reshape((b, t, h, d))?.transpose(1, 2)?; // (B,H,T,D)
        let k = self.k.forward(x)?.reshape((b, t, h, d))?.transpose(1, 2)?;
        let v = self.v.forward(x)?.reshape((b, t, h, d))?.transpose(1, 2)?;

        let q = rope.apply(&q, rope_offset, t)?;
        let k = rope.apply(&k, rope_offset, t)?;

        // Extend KV cache
        let k = if let Some(kc) = k_cache { Tensor::cat(&[kc, &k], 2)? } else { k };
        let v = if let Some(vc) = v_cache { Tensor::cat(&[vc, &v], 2)? } else { v };

        let s = k.dim(2)?; // total seq len after cache
        let mask = if causal && t > 1 {
            // Build (T, S) mask: queries attend to positions <= their own
            let m_data: Vec<f32> = (0..t * s)
                .map(|i| {
                    let q_pos = i / s + rope_offset;
                    let k_pos = i % s;
                    if k_pos <= q_pos { 0.0 } else { f32::NEG_INFINITY }
                })
                .collect();
            Some(Tensor::from_vec(m_data, (1, 1, t, s), x.device())?)
        } else {
            None
        };

        let out = sdpa(&q, &k, &v, mask.as_ref())?;
        let out = out.transpose(1, 2)?.reshape((b, t, h * d))?;
        Ok((self.o.forward(&out)?, k, v))
    }
}

// ---------------------------------------------------------------------------
// Memory Cross-Attention
// ---------------------------------------------------------------------------

pub struct MemoryAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    inv_temp: Tensor,
    n_heads: usize,
    head_dim: usize,
}

impl MemoryAttention {
    pub fn new(d_model: usize, n_heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner = n_heads * head_dim;
        let inv_temp = vb.get(n_heads, "inv_temp")?;
        Ok(Self {
            q: linear_no_bias(d_model, inner, vb.pp("q"))?,
            k: linear_no_bias(d_model, inner, vb.pp("k"))?,
            v: linear_no_bias(d_model, inner, vb.pp("v"))?,
            o: linear_no_bias(inner, d_model, vb.pp("o"))?,
            inv_temp,
            n_heads,
            head_dim,
        })
    }

    /// x: (B,T,d), mem_keys/values: (B,S,d), mem_mask: (B,S) bool as f32
    pub fn forward(&self, x: &Tensor, mem_keys: &Tensor, mem_values: &Tensor,
                   mem_mask: Option<&Tensor>) -> Result<Tensor>
    {
        let (b, t, _) = x.dims3()?;
        let s = mem_keys.dim(1)?;
        let h = self.n_heads;
        let d = self.head_dim;

        let q = self.q.forward(x)?.reshape((b, t, h, d))?.transpose(1, 2)?;
        let k = self.k.forward(mem_keys)?.reshape((b, s, h, d))?.transpose(1, 2)?;
        let v = self.v.forward(mem_values)?.reshape((b, s, h, d))?.transpose(1, 2)?;

        // Apply per-head temperature
        let temp = self.inv_temp.reshape((1, h, 1, 1))?;
        let q = q.broadcast_mul(&temp)?;

        let scale = (d as f64).sqrt();
        let kt = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn = q.matmul(&kt)?;
        let attn = (attn / scale)?;

        let attn = if let Some(mask) = mem_mask {
            // mask: (B, S) where 0=invalid -> -inf
            // Reshape to (B, 1, 1, S) for broadcasting
            let mask = mask.reshape((b, 1, 1, s))?;
            attn.broadcast_add(&mask)?
        } else {
            attn
        };

        let attn = ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v.contiguous()?)?;
        let out = out.transpose(1, 2)?.reshape((b, t, h * d))?;
        self.o.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// Transformer Block: Self-Attn → Tag-Attn → Mem-Attn → FFN
// ---------------------------------------------------------------------------

pub struct TransformerBlock {
    norm1: RMSNorm,
    attn: Attention,
    norm_tag: RMSNorm,
    tag_head: Linear,
    tag_gate: Linear,
    norm_mem: RMSNorm,
    mem_attn: MemoryAttention,
    norm2: RMSNorm,
    ffn: SiLUFFN,
}

impl TransformerBlock {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: RMSNorm::new(D_MODEL, vb.pp("norm1"))?,
            attn: Attention::new(D_MODEL, N_HEADS, HEAD_DIM, vb.pp("attn"))?,
            norm_tag: RMSNorm::new(D_MODEL, vb.pp("norm_tag"))?,
            tag_head: linear_no_bias(D_MODEL, D_MODEL, vb.pp("tag_head"))?,
            tag_gate: linear_no_bias(D_MODEL, 1, vb.pp("tag_gate"))?,
            norm_mem: RMSNorm::new(D_MODEL, vb.pp("norm_mem"))?,
            mem_attn: MemoryAttention::new(D_MODEL, N_HEADS, HEAD_DIM, vb.pp("mem_attn"))?,
            norm2: RMSNorm::new(D_MODEL, vb.pp("norm2"))?,
            ffn: SiLUFFN::new(D_MODEL, FFN_DIM, vb.pp("ffn"))?,
        })
    }

    /// Returns (x, k_cache, v_cache)
    pub fn forward(&self, x: &Tensor, rope: &RoPE, rope_offset: usize,
                   causal: bool,
                   k_cache: Option<&Tensor>, v_cache: Option<&Tensor>,
                   mem_keys: Option<&Tensor>, mem_values: Option<&Tensor>,
                   mem_mask: Option<&Tensor>,
                   tag_register: Option<&Tensor>)
        -> Result<(Tensor, Tensor, Tensor)>
    {
        // 1. Self-attention
        let (attn_out, new_k, new_v) = self.attn.forward(
            &self.norm1.forward(x)?, rope, rope_offset, causal, k_cache, v_cache)?;
        let x = (x + attn_out)?;

        // 2. Tag cross-attention (GRU-style gated update)
        let x = if let Some(tag) = tag_register {
            let normed = self.norm_tag.forward(&x)?;
            let new_tag = self.tag_head.forward(&normed)?.tanh()?;
            let gate = ops::sigmoid(&self.tag_gate.forward(&normed)?)?;
            // gate: (B, T, 1), tag: (B, d) → need to unsqueeze tag
            let tag_ctx = tag.unsqueeze(1)?; // (B, 1, d)
            let tag_update = (gate.broadcast_mul(&new_tag)?
                + (gate.broadcast_mul(&Tensor::ones_like(&gate)?)? * -1.0)?
                    .broadcast_mul(&tag_ctx)?)?;
            // Simpler: gate * new_tag + (1 - gate) * tag
            let one_minus_gate = (Tensor::ones_like(&gate)? - &gate)?;
            let tag_ctx = (gate.broadcast_mul(&new_tag)?
                + one_minus_gate.broadcast_mul(&tag_ctx)?)?;
            let _ = tag_update; // suppress unused warning
            (x + tag_ctx)?
        } else {
            x
        };

        // 3. Memory cross-attention
        let x = if let (Some(mk), Some(mv)) = (mem_keys, mem_values) {
            let mem_out = self.mem_attn.forward(&self.norm_mem.forward(&x)?, mk, mv, mem_mask)?;
            (x + mem_out)?
        } else {
            x
        };

        // 4. FFN
        let ffn_out = self.ffn.forward(&self.norm2.forward(&x)?)?;
        let x = (x + ffn_out)?;
        Ok((x, new_k, new_v))
    }
}

// ---------------------------------------------------------------------------
// KV Cache for generation
// ---------------------------------------------------------------------------

pub struct KVCache {
    pub keys: Vec<Tensor>,    // n_layers x (B, H, S, D)
    pub values: Vec<Tensor>,
    pub pos: usize,
}

impl KVCache {
    pub fn new() -> Self {
        Self { keys: Vec::new(), values: Vec::new(), pos: 0 }
    }
}

// ---------------------------------------------------------------------------
// ANT — Top-level model
// ---------------------------------------------------------------------------

pub struct ANT {
    pub embed: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RMSNorm,
    pub halt_head: Linear,
    pub addr_nets: Vec<AddrNet>,
    // v_proj uses bias (matches Python nn.Linear default)
    pub v_proj: candle_nn::Linear,
    pub rope: RoPE,
    device: Device,
}

impl ANT {
    pub fn new(vb: VarBuilder, device: &Device) -> Result<Self> {
        let embed = embedding(VOCAB_SIZE, D_MODEL, vb.pp("embed"))?;

        let mut layers = Vec::with_capacity(N_LAYERS);
        for i in 0..N_LAYERS {
            layers.push(TransformerBlock::new(vb.pp(format!("layers.{i}")))?);
        }

        let norm = RMSNorm::new(D_MODEL, vb.pp("norm"))?;
        let halt_head = candle_nn::linear(D_MODEL, 2, vb.pp("halt_head"))?;

        let mut addr_nets = Vec::with_capacity(N_ADDR_NETS);
        for i in 0..N_ADDR_NETS {
            addr_nets.push(AddrNet::new(D_MODEL, ADDR_HIDDEN_DIM, ADDR_N_BINS,
                                        vb.pp(format!("addr_nets.{i}")))?);
        }

        let v_proj = candle_nn::linear(D_MODEL, D_MODEL, vb.pp("v_proj"))?;
        let rope = RoPE::new(HEAD_DIM, MAX_SEQ_LEN, ROPE_THETA, device)?;

        Ok(Self { embed, layers, norm, halt_head, addr_nets, v_proj, rope, device: device.clone() })
    }

    /// Compute 3 hierarchical addresses from hidden state (B, d).
    /// Returns Vec of N tensors, each (B, depth) as u32.
    pub fn compute_addresses(&self, hidden: &Tensor, temperature: f32, training: bool,
                              rng: &mut impl Rng)
        -> Result<Vec<Tensor>>
    {
        let mut addrs = Vec::with_capacity(N_ADDR_NETS);
        for net in &self.addr_nets {
            let (a, _) = net.forward(hidden, temperature, training, rng)?;
            addrs.push(a);
        }
        Ok(addrs)
    }

    /// Compute addresses with logits for training losses.
    pub fn compute_addresses_with_logits(&self, hidden: &Tensor, temperature: f32,
                                          training: bool, rng: &mut impl Rng)
        -> Result<(Vec<Tensor>, Vec<Vec<Tensor>>)>
    {
        let mut addrs = Vec::new();
        let mut all_logits = Vec::new();
        for net in &self.addr_nets {
            let (a, l) = net.forward(hidden, temperature, training, rng)?;
            addrs.push(a);
            all_logits.push(l);
        }
        Ok((addrs, all_logits))
    }

    pub fn compute_value(&self, hidden: &Tensor) -> Result<Tensor> {
        self.v_proj.forward(hidden)
    }

    /// Convert (B, depth) u32 address tensors to Vec<Vec<Vec<u8>>> for trie.
    pub fn addrs_to_bytes(addr_tensors: &[Tensor]) -> Result<Vec<Vec<Vec<u8>>>> {
        let b = addr_tensors[0].dim(0)?;
        let mut batch = Vec::with_capacity(b);
        for bi in 0..b {
            let mut item_addrs = Vec::with_capacity(addr_tensors.len());
            for t in addr_tensors {
                let row = t.get(bi)?.to_vec1::<u32>()?;
                let bytes: Vec<u8> = row.iter().map(|&x| x as u8).collect();
                item_addrs.push(bytes);
            }
            batch.push(item_addrs);
        }
        Ok(batch)
    }

    /// Full forward pass.
    ///
    /// token_ids:    (B, T) u32
    /// mem_keys:     Option<(B, S, d)>
    /// mem_values:   Option<(B, S, d)>
    /// mem_mask_f32: Option<(B, S)> — 0.0 valid, -inf invalid
    /// tag_register: Option<(B, d)>
    /// kv_cache:     Option<&mut KVCache>
    ///
    /// Returns (logits: B×T×V, halt_logits: B×T×2, hidden: B×T×d)
    pub fn forward(&self, token_ids: &Tensor,
                   mem_keys: Option<&Tensor>, mem_values: Option<&Tensor>,
                   mem_mask_f32: Option<&Tensor>,
                   tag_register: Option<&Tensor>,
                   kv_cache: Option<&mut KVCache>)
        -> Result<(Tensor, Tensor, Tensor)>
    {
        let (b, t) = token_ids.dims2()?;
        let mut x = self.embed.forward(token_ids)?; // (B, T, d)

        let rope_offset = kv_cache.as_ref().map(|c| c.pos).unwrap_or(0);
        let causal = rope_offset == 0; // causal mask only on full sequence

        let mut new_keys: Vec<Tensor> = Vec::with_capacity(N_LAYERS);
        let mut new_vals: Vec<Tensor> = Vec::with_capacity(N_LAYERS);

        for (i, layer) in self.layers.iter().enumerate() {
            let (kc, vc) = if let Some(ref cache) = kv_cache {
                if i < cache.keys.len() {
                    (Some(&cache.keys[i]), Some(&cache.values[i]))
                } else { (None, None) }
            } else { (None, None) };

            let (out, nk, nv) = layer.forward(
                &x, &self.rope, rope_offset, causal, kc, vc,
                mem_keys, mem_values, mem_mask_f32, tag_register)?;
            x = out;
            new_keys.push(nk);
            new_vals.push(nv);
        }

        if let Some(cache) = kv_cache {
            cache.keys = new_keys;
            cache.values = new_vals;
            cache.pos += t;
        }

        let hidden = self.norm.forward(&x)?;
        // Weight-tied LM head: linear(hidden, embed.weight^T), reshape for 3D batch
        let embed_w = self.embed.embeddings(); // (V, d)
        let (b, t, d) = hidden.dims3()?;
        let logits = hidden
            .reshape((b * t, d))?
            .matmul(&embed_w.t()?.contiguous()?)?
            .reshape((b, t, VOCAB_SIZE as usize))?;
        let halt_logits = self.halt_head.forward(&hidden)?;

        Ok((logits, halt_logits, hidden))
    }
}
