/// ANT Engine — Per-token READ→PROCESS→WRITE cycle.
///
/// Training: two-pass encode() for correct address distribution.
/// Inference: true per-token generate() with live trie updates.

use candle_core::{Device, Result, Tensor, DType};
use rand::Rng;

use crate::config::*;
use crate::model::{ANT, KVCache};
use crate::trie::MemorySystem;

pub struct ANTEngine {
    pub model: ANT,
    pub memory: MemorySystem,
    pub device: Device,
    /// Persistent tag register: (B, d_model)
    pub tag_register: Option<Tensor>,
}

impl ANTEngine {
    pub fn new(model: ANT, mem_path: &str, device: Device) -> Self {
        let memory = MemorySystem::new(
            mem_path, D_MODEL, MEM_DEPTH_CAP,
            MEM_EMA_ALPHA_BASE, MEM_EMA_ALPHA_MIN, MEM_FLUSH_INTERVAL);
        Self { model, memory, device, tag_register: None }
    }

    pub fn reset_state(&mut self, batch_size: usize) -> Result<()> {
        self.tag_register = Some(Tensor::zeros(
            (batch_size, D_MODEL), DType::F32, &self.device)?);
        Ok(())
    }

    // ---------------------------------------------------------------------------
    // Memory I/O helpers
    // ---------------------------------------------------------------------------

    /// Convert AddrNet outputs to Vec<Vec<Vec<u8>>> for trie ops.
    fn addrs_to_bytes(addr_tensors: &[Tensor]) -> Result<Vec<Vec<Vec<u8>>>> {
        ANT::addrs_to_bytes(addr_tensors)
    }

    /// Read from trie, return (mem_keys, mem_values, mem_mask_f32).
    /// mem_keys == mem_values (trie stores single vector per node).
    /// mem_mask_f32: 0.0 = valid, -inf = invalid (for attn masking).
    pub fn read_memory(&self, hidden: &Tensor, temperature: f32, rng: &mut impl Rng)
        -> Result<(Tensor, Tensor, Tensor)>
    {
        let b = hidden.dim(0)?;
        let s = N_MEM_SLOTS;
        let d = D_MODEL;

        let addr_tensors = self.model.compute_addresses(hidden, temperature, false, rng)?;
        let batch_addrs = Self::addrs_to_bytes(&addr_tensors)?;

        let (vecs_flat, mask_flat) = self.memory.read_batch(&batch_addrs, s);

        // Build mem_keys tensor
        let vecs = Tensor::from_vec(vecs_flat, (b, s, d), &self.device)?;

        // Guard: zero out NaN/inf slots, set mask to -inf
        // For simplicity, build mask_f32 directly from bool mask
        let mask_f32_flat: Vec<f32> = mask_flat.iter().map(|&m| if m { 0.0 } else { f32::NEG_INFINITY }).collect();
        let mask_f32 = Tensor::from_vec(mask_f32_flat, (b, s), &self.device)?;

        Ok((vecs.clone(), vecs, mask_f32))
    }

    /// Write hidden states (B, d) to trie at AddrNet addresses.
    pub fn write_memory(&self, hidden: &Tensor, temperature: f32, rng: &mut impl Rng)
        -> Result<()>
    {
        let values = self.model.compute_value(hidden)?; // (B, d)
        let values_np = values.to_vec2::<f32>()?; // B x d

        // Skip any NaN/inf rows
        let valid: Vec<bool> = values_np.iter()
            .map(|row| row.iter().all(|v| v.is_finite()))
            .collect();
        if !valid.iter().any(|&v| v) { return Ok(()); }

        let filtered_hidden = if valid.iter().all(|&v| v) {
            hidden.clone()
        } else {
            // Filter to valid rows
            let indices: Vec<usize> = valid.iter().enumerate()
                .filter_map(|(i, &v)| if v { Some(i) } else { None })
                .collect();
            let rows: Vec<Tensor> = indices.iter()
                .map(|&i| hidden.get(i))
                .collect::<Result<Vec<_>>>()?;
            Tensor::stack(&rows, 0)?
        };

        let addr_tensors = self.model.compute_addresses(
            &filtered_hidden.detach(), temperature, false, rng)?;
        let batch_addrs = Self::addrs_to_bytes(&addr_tensors)?;

        let val_refs: Vec<&[f32]> = if valid.iter().all(|&v| v) {
            values_np.iter().map(|row| row.as_slice()).collect()
        } else {
            valid.iter().enumerate()
                .filter_map(|(i, &v)| if v { Some(values_np[i].as_slice()) } else { None })
                .collect()
        };
        self.memory.write_batch(&batch_addrs, &val_refs);
        Ok(())
    }

    // ---------------------------------------------------------------------------
    // Training encode — two-pass (full, single-step)
    // ---------------------------------------------------------------------------

    /// Pass-1 only: run a no-memory forward pass to compute h_mean, then read
    /// the trie and return cached memory vectors.
    ///
    /// Use this once per trie cycle before calling `forward_with_mem` in a loop.
    /// The returned tensors are detached — no gradients flow through them.
    pub fn read_for_encode(&self, token_ids: &Tensor, temperature: f32,
                           rng: &mut impl Rng)
        -> Result<(Tensor, Tensor, Tensor)>
    {
        let tag = self.tag_register.as_ref().map(|t| t.clone());
        let (_, _, hidden_p1) = self.model.forward(
            &token_ids.detach(), None, None, None, tag.as_ref(), None)?;
        let h_mean = hidden_p1.mean(1)?.detach();
        self.read_memory(&h_mean, temperature, rng)
    }

    /// Pass-2 only: forward with pre-loaded (detached) memory vectors.
    ///
    /// No trie I/O — use the tensors returned by `read_for_encode`.
    /// Updates `tag_register` with the last-token hidden state of this batch.
    pub fn forward_with_mem(&mut self, token_ids: &Tensor,
                            mem_k: &Tensor, mem_v: &Tensor, mem_mask: &Tensor)
        -> Result<(Tensor, Tensor, Tensor)>
    {
        let b = token_ids.dim(0)?;
        if self.tag_register.is_none() || self.tag_register.as_ref().unwrap().dim(0)? != b {
            self.reset_state(b)?;
        }
        let tag = self.tag_register.as_ref().map(|t| t.clone());
        let (logits, halt_logits, hidden) = self.model.forward(
            token_ids, Some(mem_k), Some(mem_v), Some(mem_mask),
            tag.as_ref(), None)?;

        // Update tag register: last-token hidden per batch item
        let mut last_rows: Vec<Tensor> = Vec::with_capacity(b);
        for bi in 0..b {
            let seq = hidden.get(bi)?;
            let last = seq.get(seq.dim(0)? - 1)?.unsqueeze(0)?;
            last_rows.push(last);
        }
        let last_hidden = Tensor::cat(&last_rows, 0)?;
        self.tag_register = Some(last_hidden.detach());

        Ok((logits, halt_logits, hidden))
    }

    /// Write a full (B, T, d) hidden tensor to the trie — one write per token.
    ///
    /// Use this once per trie cycle after the inner gradient loop completes.
    pub fn write_hidden(&self, hidden: &Tensor, temperature: f32,
                        rng: &mut impl Rng) -> Result<()>
    {
        let (b, t, _) = hidden.dims3()?;
        for ti in 0..t {
            let mut h_rows: Vec<Tensor> = Vec::with_capacity(b);
            for bi in 0..b {
                let h_t = hidden.get(bi)?.get(ti)?.unsqueeze(0)?;
                h_rows.push(h_t);
            }
            let h_batch = Tensor::cat(&h_rows, 0)?;
            self.write_memory(&h_batch.detach(), temperature, rng)?;
        }
        Ok(())
    }

    /// Two-pass training encode (single-step convenience wrapper).
    ///
    /// Pass 1: forward WITHOUT memory → get processed hidden states → addresses
    /// Pass 2: forward WITH memory (cross-attn) → logits, hidden for loss
    ///
    /// Returns (logits: B×T×V, halt_logits: B×T×2, hidden: B×T×d)
    pub fn encode(&mut self, token_ids: &Tensor, temperature: f32,
                  write_to_trie: bool, rng: &mut impl Rng)
        -> Result<(Tensor, Tensor, Tensor)>
    {
        let b = token_ids.dim(0)?;

        if self.tag_register.is_none() || self.tag_register.as_ref().unwrap().dim(0)? != b {
            self.reset_state(b)?;
        }

        let tag = self.tag_register.as_ref().map(|t| t.clone());

        // Pass 1: no memory
        let (_, _, hidden_p1) = {
            let no_grad_ids = token_ids.detach();
            self.model.forward(&no_grad_ids, None, None, None, tag.as_ref(), None)?
        };

        // Mean pool over sequence for address generation
        let h_mean = hidden_p1.mean(1)?; // (B, d)

        // Read memory using pass-1 hidden
        let (mem_k, mem_v, mem_mask) = self.read_memory(&h_mean.detach(), temperature, rng)?;

        // Pass 2: with memory
        let (logits, halt_logits, hidden) = self.model.forward(
            token_ids,
            Some(&mem_k), Some(&mem_v), Some(&mem_mask),
            tag.as_ref(), None)?;

        // Update tag register: last token hidden for each batch item
        let mut last_rows: Vec<Tensor> = Vec::with_capacity(b);
        for bi in 0..b {
            let seq = hidden.get(bi)?; // (T, d)
            let last = seq.get(seq.dim(0)? - 1)?; // (d,)
            last_rows.push(last.unsqueeze(0)?); // (1, d)
        }
        let last_hidden = Tensor::cat(&last_rows, 0)?; // (B, d)
        self.tag_register = Some(last_hidden.detach());

        // Write to trie
        if write_to_trie {
            let t = hidden.dim(1)?;
            for ti in 0..t {
                let mut h_rows: Vec<Tensor> = Vec::with_capacity(b);
                for bi in 0..b {
                    let h_t = hidden.get(bi)?.get(ti)?.unsqueeze(0)?; // (1, d)
                    h_rows.push(h_t);
                }
                let h_batch = Tensor::cat(&h_rows, 0)?; // (B, d)
                self.write_memory(&h_batch.detach(), temperature, rng)?;
            }
        }

        Ok((logits, halt_logits, hidden))
    }

    // ---------------------------------------------------------------------------
    // Inference generate
    // ---------------------------------------------------------------------------

    pub fn generate(&mut self, prompt_ids: &[u32], max_tokens: usize,
                    temperature: f32, top_k: usize, top_p: f32,
                    rng: &mut impl Rng) -> Result<Vec<u32>>
    {
        self.reset_state(1)?;
        let device = &self.device.clone();

        let prompt = Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_ids.len()), device)?;

        // Prefill: two-pass
        let (_, _, h_pre) = self.model.forward(&prompt, None, None, None,
                                               self.tag_register.as_ref(), None)?;
        let h_mean = h_pre.mean(1)?;
        let (mk, mv, mm) = self.read_memory(&h_mean, temperature, rng)?;

        let mut cache = KVCache::new();
        let (logits, _, hidden) = self.model.forward(
            &prompt, Some(&mk), Some(&mv), Some(&mm),
            self.tag_register.as_ref(), Some(&mut cache))?;

        // Write prompt tokens to trie
        let t = hidden.dim(1)?;
        for ti in 0..t {
            let h_t = hidden.get(0)?.get(ti)?.unsqueeze(0)?;
            self.write_memory(&h_t, temperature, rng)?;
        }

        let tag_last = hidden.get(0)?.get(t - 1)?.unsqueeze(0)?.detach();
        self.tag_register = Some(tag_last);

        // Sample first token
        let next_logits = logits.get(0)?.get(logits.dim(1)? - 1)?.unsqueeze(0)?;
        let mut next_token = self.sample(&next_logits, temperature, top_k, top_p, rng)?;
        let mut generated = vec![next_token];
        let mut prev_hidden = hidden.get(0)?.get(t - 1)?.unsqueeze(0)?;

        for _ in 1..max_tokens {
            if next_token == EOS_ID { break; }

            let tok = Tensor::from_vec(vec![next_token], (1, 1), device)?;
            let (mk, mv, mm) = self.read_memory(&prev_hidden, temperature, rng)?;

            let (logits, halt_logits, hidden) = self.model.forward(
                &tok, Some(&mk), Some(&mv), Some(&mm),
                self.tag_register.as_ref(), Some(&mut cache))?;

            // Write to trie
            let h0 = hidden.get(0)?.get(0)?.unsqueeze(0)?;
            self.write_memory(&h0.detach(), temperature, rng)?;

            // Update tag register
            self.tag_register = Some(h0.detach());

            // Halt head: check if model wants more memory cycles
            let halt_prob = candle_nn::ops::softmax(&halt_logits.get(0)?.get(0)?, 0)?;
            let halt = halt_prob.get(1)?.to_scalar::<f32>()? > 0.5;

            if !halt {
                // Up to 3 extra memory fetch cycles
                for _ in 0..3 {
                    let (mk2, mv2, mm2) = self.read_memory(
                        &hidden.get(0)?.get(0)?.unsqueeze(0)?, temperature, rng)?;
                    let _ = self.model.forward(
                        &tok, Some(&mk2), Some(&mv2), Some(&mm2),
                        self.tag_register.as_ref(), None)?;
                }
            }

            prev_hidden = hidden.get(0)?.get(0)?.unsqueeze(0)?;
            let nl = logits.get(0)?.get(0)?.unsqueeze(0)?;
            next_token = self.sample(&nl, temperature, top_k, top_p, rng)?;

            if next_token == NOOP_ID { continue; }
            generated.push(next_token);
        }

        Ok(generated)
    }

    fn sample(&self, logits: &Tensor, temperature: f32, top_k: usize, top_p: f32,
              rng: &mut impl Rng) -> Result<u32>
    {
        let mut logits = logits.squeeze(0)?.to_vec1::<f32>()?;

        // Check for NaN/inf
        if logits.iter().any(|v| !v.is_finite()) {
            return Ok(UNK_ID);
        }

        if temperature <= 0.0 {
            return Ok(logits.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap_or(UNK_ID));
        }

        // Temperature scaling
        for v in logits.iter_mut() { *v /= temperature; }

        // Top-k
        if top_k > 0 && top_k < logits.len() {
            let mut sorted = logits.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let threshold = sorted[top_k - 1];
            for v in logits.iter_mut() {
                if *v < threshold { *v = f32::NEG_INFINITY; }
            }
        }

        // Softmax
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        let mut probs: Vec<f32> = exp.iter().map(|&v| v / sum).collect();

        // Top-p nucleus
        if top_p < 1.0 {
            let mut pairs: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut cum = 0.0;
            for (_, p) in &pairs {
                cum += p;
                if cum >= top_p { break; }
            }
            let threshold = pairs.iter().scan(0.0f32, |acc, (_, p)| {
                *acc += p; Some(*acc)
            }).zip(pairs.iter())
              .find(|(c, _)| *c >= top_p)
              .map(|(_, (_, p))| *p)
              .unwrap_or(0.0);
            for (i, p) in probs.iter_mut().enumerate() {
                if pairs.iter().position(|(pi, _)| *pi == i)
                       .map(|pos| {
                           let cum: f32 = pairs[..pos].iter().map(|(_, p)| p).sum();
                           cum >= top_p
                       })
                       .unwrap_or(false)
                {
                    *p = 0.0;
                }
            }
            let _ = threshold;
            let sum: f32 = probs.iter().sum();
            if sum > 0.0 { for p in probs.iter_mut() { *p /= sum; } }
        }

        // Sample
        let r: f32 = rng.gen();
        let mut cum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if r <= cum { return Ok(i as u32); }
        }
        Ok((probs.len() - 1) as u32)
    }

    pub fn flush(&self) { self.memory.flush(); }

    pub fn reset_memory(&self) { self.memory.reset(); }

    pub fn memory_stats(&self) -> (usize, u64) {
        (self.memory.total_nodes(), self.memory.total_entries())
    }
}
