/// ANT Training — Phase A/B/C curriculum.

use candle_core::{Device, Result, Tensor, DType, Var};
use candle_nn::{VarMap, Optimizer};
use candle_nn::optim::AdamW;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::time::Instant;

use crate::config::*;
use crate::model::ANT;
use crate::engine::ANTEngine;
use crate::data::{TextDataset, random_batch, load_all_texts, shell_texts, DataSplit};

// ---------------------------------------------------------------------------
// Parameter group tracking for curriculum freezing
// ---------------------------------------------------------------------------

/// Three named VarMaps that together hold all model parameters.
///
/// Phase A: only base_vm trained   (addr/mem receive no gradients — no memory ops)
/// Phase B: only addr_vm + mem_vm  (base_vm frozen — learn to address and read/write)
/// Phase C: only base_vm + mem_vm  (addr_vm frozen — base learns to USE memory)
pub struct ModelVarMaps {
    pub base: VarMap,   // embed, self-attn, FFN, final norm
    pub addr: VarMap,   // addr_nets (×3), v_proj
    pub mem:  VarMap,   // mem_attn, tag system, halt_head
}

impl ModelVarMaps {
    pub fn new() -> Self {
        Self { base: VarMap::new(), addr: VarMap::new(), mem: VarMap::new() }
    }

    pub fn base_vars(&self)         -> Vec<Var> { self.base.all_vars() }
    pub fn addr_and_mem_vars(&self) -> Vec<Var> {
        let mut v = self.addr.all_vars(); v.extend(self.mem.all_vars()); v
    }
    pub fn base_and_mem_vars(&self) -> Vec<Var> {
        let mut v = self.base.all_vars(); v.extend(self.mem.all_vars()); v
    }
    pub fn all_vars(&self) -> Vec<Var> {
        let mut v = self.base.all_vars();
        v.extend(self.addr.all_vars());
        v.extend(self.mem.all_vars());
        v
    }

    pub fn count_params(vars: &[Var]) -> usize {
        vars.iter().map(|v| v.as_tensor().elem_count()).sum()
    }

    pub fn save(&self, step: usize, phase: &str, ckpt_dir: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(ckpt_dir)?;
        let path   = format!("{}/checkpoint_phase{}.safetensors", ckpt_dir, phase);
        let latest = format!("{}/checkpoint_latest.safetensors",   ckpt_dir);

        let mut data: std::collections::HashMap<String, candle_core::Tensor> =
            std::collections::HashMap::new();
        for (key, var) in self.base.data().lock().unwrap().iter() {
            data.insert(key.clone(), var.as_tensor().clone());
        }
        for (key, var) in self.addr.data().lock().unwrap().iter() {
            data.insert(key.clone(), var.as_tensor().clone());
        }
        for (key, var) in self.mem.data().lock().unwrap().iter() {
            data.insert(key.clone(), var.as_tensor().clone());
        }

        if let Err(e) = candle_core::safetensors::save(&data, &path) {
            eprintln!("  Warning: failed to save {}: {}", path, e);
        } else {
            println!("  Saved checkpoint: {} ({} tensors, step {})", path, data.len(), step);
        }
        let _ = std::fs::copy(&path, &latest);
        Ok(())
    }
}


// ---------------------------------------------------------------------------
// Cross-entropy loss (manual, since candle-nn doesn't have a built-in with pad)
// ---------------------------------------------------------------------------

/// Cross-entropy loss over (B*T, V) logits and (B*T,) targets, ignoring pad_id.
fn cross_entropy_with_ignore(logits: &Tensor, targets: &Tensor, ignore_id: u32) -> Result<Tensor> {
    let (bt, v) = logits.dims2()?;
    let targets_flat = targets.flatten_all()?;

    // Build valid mask
    let target_vec = targets_flat.to_vec1::<u32>()?;
    let mask: Vec<f32> = target_vec.iter()
        .map(|&t| if t == ignore_id { 0.0 } else { 1.0 })
        .collect();

    let n_valid = mask.iter().filter(|&&v| v > 0.0).count().max(1);
    let mask_t = Tensor::from_vec(mask, bt, logits.device())?;

    // log_softmax
    let log_probs = candle_nn::ops::log_softmax(logits, 1)?;

    // Gather log probs at target indices, zeroing pad
    // Replace pad targets with 0 to avoid out-of-bounds
    let safe_targets: Vec<u32> = target_vec.iter()
        .map(|&t| if t == ignore_id { 0 } else { t })
        .collect();
    let target_idx = Tensor::from_vec(safe_targets, bt, logits.device())?;

    // gather: (BT, 1)
    let gathered = log_probs.gather(&target_idx.unsqueeze(1)?, 1)?.squeeze(1)?;
    let masked = gathered.mul(&mask_t)?;
    let loss = (masked.sum_all()?.neg()? * (1.0 / n_valid as f64))?;
    Ok(loss)
}

// ---------------------------------------------------------------------------
// Checkpoint I/O — kept for external callers; delegates to ModelVarMaps.save
// ---------------------------------------------------------------------------

pub fn save_checkpoint(vms: &ModelVarMaps, step: usize, phase: &str, ckpt_dir: &str) -> std::io::Result<()> {
    vms.save(step, phase, ckpt_dir)
}

// ---------------------------------------------------------------------------
// Losses
// ---------------------------------------------------------------------------

/// KL divergence loss between two log-softmax distributions.
fn kl_div_loss(log_p: &Tensor, q: &Tensor) -> Result<Tensor> {
    let kl = (q * (q.log()? - log_p)?)?;
    kl.sum_all()? / Tensor::from_vec(vec![q.dim(0)? as f32], 1, q.device())?.squeeze(0)?
}

/// Contrastive address loss: two passes of compute_addresses on same hidden → KL divergence.
fn contrastive_address_loss(
    logits_a: &[Vec<Tensor>],
    logits_b: &[Vec<Tensor>],
) -> Result<Tensor> {
    let mut losses: Vec<Tensor> = Vec::new();
    for (net_a, net_b) in logits_a.iter().zip(logits_b.iter()) {
        for (la, lb) in net_a.iter().zip(net_b.iter()) {
            let log_pa = candle_nn::ops::log_softmax(la, 1)?;
            let pb = candle_nn::ops::softmax(lb, 1)?;
            losses.push(kl_div_loss(&log_pa, &pb)?);
        }
    }
    if losses.is_empty() {
        return Tensor::zeros((), DType::F32, logits_a[0][0].device());
    }
    let sum = losses.iter().skip(1).fold(losses[0].clone(), |acc, t| {
        acc.add(t).unwrap_or(acc)
    });
    sum / Tensor::from_vec(vec![losses.len() as f32], 1, logits_a[0][0].device())?.squeeze(0)?
}

/// Quadratic depth cost: higher entropy at deeper trie levels is penalised more.
fn depth_cost(logits_list: &[Vec<Tensor>], penalty_scale: f32) -> Result<Tensor> {
    let dev = logits_list[0][0].device();
    let mut losses: Vec<Tensor> = Vec::new();
    for net_logits in logits_list {
        for (d, logits) in net_logits.iter().enumerate() {
            let probs = candle_nn::ops::softmax(logits, 1)?;
            let log_probs = probs.log()?;
            let entropy = (probs.mul(&log_probs)?)
                .sum_keepdim(1)?
                .neg()?
                .mean_all()?;
            let weight = ((d + 1) as f32).powi(2);
            losses.push((entropy * (weight as f64))?);
        }
    }
    if losses.is_empty() {
        return Tensor::zeros((), DType::F32, dev);
    }
    let sum = losses.iter().skip(1).fold(losses[0].clone(), |acc, t| {
        acc.add(t).unwrap_or(acc)
    });
    let avg = (sum * (1.0 / losses.len() as f64))?;
    Ok((avg * penalty_scale as f64)?)
}

// ---------------------------------------------------------------------------
// Phase A — Pure LM (no memory)
// ---------------------------------------------------------------------------

pub struct TrainConfig {
    pub steps_a: usize,
    pub steps_b: usize,
    pub steps_c: usize,
    pub lr_a: f64,
    pub lr_b: f64,
    pub lr_c: f64,
    pub batch_a: usize,
    pub batch_b: usize,
    pub batch_c: usize,
    /// Gradient steps per trie read/write cycle (Phases B and C).
    /// Effective batch = batch_b * inner_steps. Default mirrors INNER_STEPS constant.
    pub inner_steps: usize,
    pub data_dir: String,
    pub ckpt_dir: String,
    pub log_every: usize,
    pub save_every: usize,
    pub max_seq_len: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps_a: 5000,
            steps_b: 3000,
            steps_c: 5000,
            lr_a: 3e-4,
            lr_b: 1e-3,
            lr_c: 1e-4,
            batch_a: 16,
            batch_b: 8,
            batch_c: 8,
            inner_steps: INNER_STEPS,
            data_dir: "data_cache".to_string(),
            ckpt_dir: "checkpoints/rust".to_string(),
            log_every: 50,
            save_every: 500,
            max_seq_len: MAX_SEQ_LEN,
        }
    }
}

pub fn phase_a(engine: &mut ANTEngine, cfg: &TrainConfig, vms: &ModelVarMaps,
               device: &Device) -> Result<()>
{
    println!("\n{}", "=".repeat(60));
    println!("  Phase A — Pure LM (no memory) | steps 0→{}", cfg.steps_a);
    // Phase A: only base params trained; addr/mem have no gradient path.
    let params = vms.base_vars();
    println!("  Trainable params: {} (base only)", ModelVarMaps::count_params(&params));
    println!("{}\n", "=".repeat(60));

    let mut texts = load_all_texts(&cfg.data_dir);
    texts.extend(shell_texts());
    println!("  {} texts loaded", texts.len());

    let split = DataSplit::from_texts(texts, 0.05);
    let dataset = TextDataset::new(split.train, cfg.max_seq_len);

    let params: Vec<Var> = params;
    let adamw_cfg = candle_nn::optim::ParamsAdamW {
        lr: cfg.lr_a,
        weight_decay: 0.01,
        ..Default::default()
    };
    let mut opt = AdamW::new(params, adamw_cfg)?;

    let mut rng = StdRng::seed_from_u64(42);
    let mut step = 0usize;
    let mut running_loss = 0.0f32;
    let mut t0 = Instant::now();

    while step < cfg.steps_a {
        let (batch_flat, b, t) = random_batch(&dataset, cfg.batch_a, &mut rng);

        // Per-row shift: input = row[0..t-1], target = row[1..t]
        let t1 = t.saturating_sub(1);
        let mut input_vec: Vec<u32> = Vec::with_capacity(b * t1);
        let mut target_vec: Vec<u32> = Vec::with_capacity(b * t1);
        for bi in 0..b {
            input_vec.extend_from_slice(&batch_flat[bi*t .. bi*t + t1]);
            target_vec.extend_from_slice(&batch_flat[bi*t + 1 .. bi*t + t]);
        }
        let input = Tensor::from_vec(input_vec, (b, t1), device)?;
        let target = Tensor::from_vec(target_vec, (b, t1), device)?;

        // Pure forward — no memory
        let (logits, _halt, _hidden) = engine.model.forward(
            &input, None, None, None, None, None)?;

        let bt = b * t1;
        let logits_2d = logits.reshape((bt, VOCAB_SIZE))?;
        let target_flat = target.flatten_all()?;
        let loss = cross_entropy_with_ignore(&logits_2d, &target_flat, PAD_ID as u32)?;

        let loss_val = loss.to_scalar::<f32>()?;
        if loss_val.is_nan() || loss_val.is_infinite() {
            eprintln!("  NaN loss at step {}, skipping", step);
            step += 1;
            continue;
        }

        opt.backward_step(&loss)?;

        step += 1;
        running_loss += loss_val;

        if step % cfg.log_every == 0 {
            let avg = running_loss / cfg.log_every as f32;
            let elapsed = t0.elapsed().as_secs_f32();
            let its = cfg.log_every as f32 / elapsed;
            println!("  A step {}/{} | loss {:.4} | {:.1} it/s", step, cfg.steps_a, avg, its);
            running_loss = 0.0;
            t0 = Instant::now();
        }

        if step % cfg.save_every == 0 {
            let _ = save_checkpoint(vms, step, "A_latest", &cfg.ckpt_dir);
        }
    }

    let _ = save_checkpoint(vms, step, "A", &cfg.ckpt_dir);
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase B — Memory training (frozen base)
// ---------------------------------------------------------------------------

pub fn phase_b(engine: &mut ANTEngine, cfg: &TrainConfig, vms: &ModelVarMaps,
               device: &Device) -> Result<()>
{
    println!("\n{}", "=".repeat(60));
    println!("  Phase B — Memory Training (frozen base) | steps 0→{}", cfg.steps_b);
    // Phase B: freeze base; train AddrNets, v_proj, mem_attn, tags, halt_head.
    let params = vms.addr_and_mem_vars();
    println!("  Trainable params: {} (addr + mem) | frozen: {} (base)",
        ModelVarMaps::count_params(&params),
        ModelVarMaps::count_params(&vms.base_vars()));
    println!("{}\n", "=".repeat(60));

    // Reset trie for clean Phase B start
    engine.reset_memory();
    println!("  Trie reset for Phase B");

    let adamw_cfg = candle_nn::optim::ParamsAdamW {
        lr: cfg.lr_b,
        weight_decay: 0.01,
        ..Default::default()
    };
    let mut opt = AdamW::new(params, adamw_cfg)?;

    let mut texts = load_all_texts(&cfg.data_dir);
    texts.extend(shell_texts());
    let split = DataSplit::from_texts(texts, 0.05);
    let dataset = TextDataset::new(split.train, cfg.max_seq_len);

    let mut rng = StdRng::seed_from_u64(123);
    let mut step = 0usize;
    let mut running_lm = 0.0f32;
    let mut running_contr = 0.0f32;
    let mut running_depth = 0.0f32;
    let mut t0 = Instant::now();

    while step < cfg.steps_b {
        let temperature = (1.0f32 - step as f32 / cfg.steps_b as f32).max(0.5);

        // ── Trie read: one pass-1 forward + trie read shared across inner steps ──
        let (seed_flat, seed_b, seed_t) = random_batch(&dataset, cfg.batch_b, &mut rng);
        let seed_t1 = seed_t.saturating_sub(1);
        let mut seed_in: Vec<u32> = Vec::with_capacity(seed_b * seed_t1);
        for bi in 0..seed_b {
            seed_in.extend_from_slice(&seed_flat[bi*seed_t .. bi*seed_t + seed_t1]);
        }
        let seed_input = Tensor::from_vec(seed_in, (seed_b, seed_t1), device)?;
        engine.reset_state(seed_b)?;
        let (mem_k, mem_v, mem_mask) = engine.read_for_encode(&seed_input, temperature, &mut rng)?;

        // ── Inner gradient loop: INNER_STEPS mini-batches share the same mem_vecs ──
        let mut accumulated_loss: Option<Tensor> = None;
        let mut last_hidden: Option<Tensor> = None;
        let mut inner_lm = 0.0f32;
        let mut inner_contr = 0.0f32;
        let mut inner_depth = 0.0f32;
        let scale = 1.0 / cfg.inner_steps as f64;

        for _inner in 0..cfg.inner_steps {
            let (batch_flat, b, t) = random_batch(&dataset, cfg.batch_b, &mut rng);
            let t1 = t.saturating_sub(1);
            let mut input_vec: Vec<u32> = Vec::with_capacity(b * t1);
            let mut target_vec: Vec<u32> = Vec::with_capacity(b * t1);
            for bi in 0..b {
                input_vec.extend_from_slice(&batch_flat[bi*t .. bi*t + t1]);
                target_vec.extend_from_slice(&batch_flat[bi*t + 1 .. bi*t + t]);
            }
            let input  = Tensor::from_vec(input_vec,  (b, t1), device)?;
            let target = Tensor::from_vec(target_vec, (b, t1), device)?;

            let (logits, _halt, hidden) = engine.forward_with_mem(
                &input, &mem_k, &mem_v, &mem_mask)?;

            let bt = b * t1;
            let logits_2d  = logits.reshape((bt, VOCAB_SIZE))?;
            let target_flat = target.flatten_all()?;
            let l_lm = cross_entropy_with_ignore(&logits_2d, &target_flat, PAD_ID as u32)?;

            let h_mean = hidden.mean(1)?;
            let (_, addr_logits_a) = engine.model.compute_addresses_with_logits(
                &h_mean, temperature, false, &mut rng)?;
            let (_, addr_logits_b) = engine.model.compute_addresses_with_logits(
                &h_mean, temperature, false, &mut rng)?;

            let l_contrastive = contrastive_address_loss(&addr_logits_a, &addr_logits_b)?;
            let l_depth = depth_cost(&addr_logits_a, 0.01)?;

            inner_lm    += l_lm.to_scalar::<f32>()?;
            inner_contr += l_contrastive.to_scalar::<f32>()?;
            inner_depth += l_depth.to_scalar::<f32>()?;

            let inner_loss = ((l_lm + (l_contrastive * 0.1)?)? + l_depth)?;
            let inner_loss_scaled = (inner_loss * scale)?;

            accumulated_loss = Some(match accumulated_loss {
                None      => inner_loss_scaled,
                Some(acc) => (acc + inner_loss_scaled)?,
            });
            last_hidden = Some(hidden);
        }

        let total_loss = accumulated_loss.unwrap();
        let loss_val = total_loss.to_scalar::<f32>()?;
        if loss_val.is_nan() || loss_val.is_infinite() {
            eprintln!("  NaN/inf loss at step {}, skipping", step);
            step += 1;
            continue;
        }

        opt.backward_step(&total_loss)?;

        // ── Trie write: once per cycle using the last inner step's hidden states ──
        if let Some(h) = last_hidden {
            engine.write_hidden(&h, temperature, &mut rng)?;
        }

        step += 1;
        running_lm    += inner_lm    / cfg.inner_steps as f32;
        running_contr += inner_contr / cfg.inner_steps as f32;
        running_depth += inner_depth / cfg.inner_steps as f32;

        if step % cfg.log_every == 0 {
            let elapsed = t0.elapsed().as_secs_f32();
            let its = cfg.log_every as f32 / elapsed;
            let (nodes, entries) = engine.memory_stats();
            println!(
                "  B step {}/{} | lm {:.4} contr {:.4} depth {:.4} | {:.1} it/s ({} inner) | trie {} nodes",
                step, cfg.steps_b,
                running_lm    / cfg.log_every as f32,
                running_contr / cfg.log_every as f32,
                running_depth / cfg.log_every as f32,
                its, cfg.inner_steps, nodes
            );
            running_lm = 0.0; running_contr = 0.0; running_depth = 0.0;
            t0 = Instant::now();
        }

        if step % cfg.save_every == 0 {
            let _ = save_checkpoint(vms, step, "B_latest", &cfg.ckpt_dir);
            engine.flush();
        }
    }

    let _ = save_checkpoint(vms, step, "B", &cfg.ckpt_dir);
    engine.flush();
    Ok(())
}

// ---------------------------------------------------------------------------
// Phase C — End-to-end with memory
// ---------------------------------------------------------------------------

pub fn phase_c(engine: &mut ANTEngine, cfg: &TrainConfig, vms: &ModelVarMaps,
               device: &Device) -> Result<()>
{
    println!("\n{}", "=".repeat(60));
    println!("  Phase C — End-to-End with Memory | steps 0→{}", cfg.steps_c);
    // Phase C: freeze addr (AddrNets + v_proj); train base + mem components.
    let params = vms.base_and_mem_vars();
    println!("  Trainable params: {} (base + mem) | frozen: {} (addr)",
        ModelVarMaps::count_params(&params),
        ModelVarMaps::count_params(&vms.addr.all_vars()));
    println!("{}\n", "=".repeat(60));

    let adamw_cfg = candle_nn::optim::ParamsAdamW {
        lr: cfg.lr_c,
        weight_decay: 0.01,
        ..Default::default()
    };
    let mut opt = AdamW::new(params, adamw_cfg)?;

    let mut texts = load_all_texts(&cfg.data_dir);
    texts.extend(shell_texts());
    let split = DataSplit::from_texts(texts, 0.05);
    let dataset = TextDataset::new(split.train, cfg.max_seq_len);

    let mut rng = StdRng::seed_from_u64(456);
    let mut step = 0usize;
    let mut running_loss = 0.0f32;
    let mut t0 = Instant::now();

    while step < cfg.steps_c {
        // ── Trie read: shared across inner steps ──
        let (seed_flat, seed_b, seed_t) = random_batch(&dataset, cfg.batch_c, &mut rng);
        let seed_t1 = seed_t.saturating_sub(1);
        let mut seed_in: Vec<u32> = Vec::with_capacity(seed_b * seed_t1);
        for bi in 0..seed_b {
            seed_in.extend_from_slice(&seed_flat[bi*seed_t .. bi*seed_t + seed_t1]);
        }
        let seed_input = Tensor::from_vec(seed_in, (seed_b, seed_t1), device)?;
        engine.reset_state(seed_b)?;
        let (mem_k, mem_v, mem_mask) = engine.read_for_encode(&seed_input, 1.0, &mut rng)?;

        // ── Inner gradient loop ──
        let mut accumulated_loss: Option<Tensor> = None;
        let mut last_hidden: Option<Tensor> = None;
        let scale = 1.0 / cfg.inner_steps as f64;

        for _inner in 0..cfg.inner_steps {
            let (batch_flat, b, t) = random_batch(&dataset, cfg.batch_c, &mut rng);
            let t1 = t.saturating_sub(1);
            let mut input_vec: Vec<u32> = Vec::with_capacity(b * t1);
            let mut target_vec: Vec<u32> = Vec::with_capacity(b * t1);
            for bi in 0..b {
                input_vec.extend_from_slice(&batch_flat[bi*t .. bi*t + t1]);
                target_vec.extend_from_slice(&batch_flat[bi*t + 1 .. bi*t + t]);
            }
            let input  = Tensor::from_vec(input_vec,  (b, t1), device)?;
            let target = Tensor::from_vec(target_vec, (b, t1), device)?;

            let (logits, _halt, hidden) = engine.forward_with_mem(
                &input, &mem_k, &mem_v, &mem_mask)?;

            let bt = b * t1;
            let logits_2d  = logits.reshape((bt, VOCAB_SIZE))?;
            let target_flat = target.flatten_all()?;
            let l = cross_entropy_with_ignore(&logits_2d, &target_flat, PAD_ID as u32)?;

            running_loss += l.to_scalar::<f32>()?;

            let l_scaled = (l * scale)?;
            accumulated_loss = Some(match accumulated_loss {
                None      => l_scaled,
                Some(acc) => (acc + l_scaled)?,
            });
            last_hidden = Some(hidden);
        }

        let total_loss = accumulated_loss.unwrap();
        let loss_val = total_loss.to_scalar::<f32>()?;
        if loss_val.is_nan() || loss_val.is_infinite() {
            eprintln!("  NaN/inf loss at step {}, skipping", step);
            step += 1;
            continue;
        }

        opt.backward_step(&total_loss)?;

        // ── Trie write: once per cycle ──
        if let Some(h) = last_hidden {
            engine.write_hidden(&h, 1.0, &mut rng)?;
        }

        step += 1;

        if step % cfg.log_every == 0 {
            let elapsed = t0.elapsed().as_secs_f32();
            let its = cfg.log_every as f32 / elapsed;
            let (nodes, _) = engine.memory_stats();
            println!(
                "  C step {}/{} | loss {:.4} | {:.1} it/s ({} inner) | trie {} nodes",
                step, cfg.steps_c,
                running_loss / (cfg.log_every * cfg.inner_steps) as f32,
                its, cfg.inner_steps, nodes
            );
            running_loss = 0.0;
            t0 = Instant::now();
        }

        if step % cfg.save_every == 0 {
            let _ = save_checkpoint(vms, step, "C_latest", &cfg.ckpt_dir);
            engine.flush();
        }
    }

    let _ = save_checkpoint(vms, step, "C", &cfg.ckpt_dir);
    engine.flush();
    Ok(())
}

// ---------------------------------------------------------------------------
// Run full curriculum
// ---------------------------------------------------------------------------

pub fn run(cfg: TrainConfig, device: Device) -> Result<()> {
    let vms = ModelVarMaps::new();
    let model = ANT::new_split(&vms.base, &vms.addr, &vms.mem, &device)?;
    let mem_path = format!("{}/trie", cfg.ckpt_dir);
    let mut engine = ANTEngine::new(model, &mem_path, device.clone());

    println!("  ANT Rust training | device: {:?}", device);
    let total = ModelVarMaps::count_params(&vms.all_vars());
    let base_n = ModelVarMaps::count_params(&vms.base_vars());
    let addr_n = ModelVarMaps::count_params(&vms.addr.all_vars());
    let mem_n  = ModelVarMaps::count_params(&vms.mem.all_vars());
    println!("  Total params: {} (base={} addr={} mem={})", total, base_n, addr_n, mem_n);

    phase_a(&mut engine, &cfg, &vms, &device)?;
    phase_b(&mut engine, &cfg, &vms, &device)?;
    phase_c(&mut engine, &cfg, &vms, &device)?;

    println!("\nTraining complete.");
    Ok(())
}
