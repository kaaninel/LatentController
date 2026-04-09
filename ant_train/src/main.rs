/// ANT — Rust/Candle training binary entry point.
///
/// Usage:
///   cargo run --bin train [-- --device cpu|cuda|metal] [--steps-a N] ...

mod config;
mod trie;
mod model;
mod engine;
mod data;
mod train;
mod inference;

use std::env;
use candle_core::Device;

fn parse_device(args: &[String]) -> Device {
    for i in 0..args.len() {
        if args[i] == "--device" && i + 1 < args.len() {
            match args[i + 1].as_str() {
                "cuda" => {
                    #[cfg(feature = "cuda")]
                    return Device::new_cuda(0).unwrap_or_else(|e| {
                        eprintln!("CUDA not available: {}. Falling back to CPU.", e);
                        Device::Cpu
                    });
                    #[cfg(not(feature = "cuda"))]
                    eprintln!("CUDA feature not compiled. Run: cargo run --features cuda");
                }
                "metal" => {
                    #[cfg(feature = "metal")]
                    return Device::new_metal(0).unwrap_or_else(|e| {
                        eprintln!("Metal not available: {}. Falling back to CPU.", e);
                        Device::Cpu
                    });
                    #[cfg(not(feature = "metal"))]
                    eprintln!("Metal feature not compiled. Run: cargo run --features metal");
                }
                "cpu" => return Device::Cpu,
                other => eprintln!("Unknown device '{}', using CPU", other),
            }
        }
    }
    Device::Cpu
}

fn parse_usize(args: &[String], flag: &str, default: usize) -> usize {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse::<usize>() {
                return v;
            }
        }
    }
    default
}

fn parse_f64(args: &[String], flag: &str, default: f64) -> f64 {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            if let Ok(v) = args[i + 1].parse::<f64>() {
                return v;
            }
        }
    }
    default
}

fn parse_str<'a>(args: &'a [String], flag: &str, default: &'a str) -> String {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return args[i + 1].clone();
        }
    }
    default.to_string()
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();

    let device = parse_device(&args);
    println!("Device: {:?}", device);

    let cfg = train::TrainConfig {
        steps_a: parse_usize(&args, "--steps-a", 5000),
        steps_b: parse_usize(&args, "--steps-b", 3000),
        steps_c: parse_usize(&args, "--steps-c", 5000),
        lr_a: parse_f64(&args, "--lr-a", 3e-4),
        lr_b: parse_f64(&args, "--lr-b", 1e-3),
        lr_c: parse_f64(&args, "--lr-c", 1e-4),
        batch_a: parse_usize(&args, "--batch-a", 16),
        batch_b: parse_usize(&args, "--batch-b", 8),
        batch_c: parse_usize(&args, "--batch-c", 8),
        inner_steps: parse_usize(&args, "--inner-steps", config::INNER_STEPS),
        data_dir: parse_str(&args, "--data-dir", "data_cache"),
        ckpt_dir: parse_str(&args, "--ckpt-dir", "checkpoints/rust"),
        log_every: parse_usize(&args, "--log-every", 50),
        save_every: parse_usize(&args, "--save-every", 500),
        max_seq_len: parse_usize(&args, "--max-seq-len", config::MAX_SEQ_LEN),
    };

    if let Err(e) = train::run(cfg, device) {
        eprintln!("Training error: {}", e);
        std::process::exit(1);
    }
}
