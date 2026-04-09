/// ANT chat inference binary.
///
/// Usage: cargo run --bin chat [-- --mem <path> --ckpt <path>]

use candle_core::{Device, DType};
use candle_nn::{VarMap, VarBuilder};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::io::{self, Write, BufRead};

use crate::config::*;
use crate::model::ANT;
use crate::engine::ANTEngine;
use crate::data::detokenize;

pub fn run_chat(ckpt_path: Option<&str>, mem_path: &str, device: Device) -> candle_core::Result<()> {
    let mut var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, DType::F32, &device);

    let model = ANT::new(vb, &device)?;

    // Load checkpoint if provided
    if let Some(path) = ckpt_path {
        if std::path::Path::new(path).exists() {
            println!("Loading checkpoint: {}", path);
            var_map.load(path)?;
        } else {
            eprintln!("Warning: checkpoint not found at {}, using random weights", path);
        }
    }

    let mut engine = ANTEngine::new(model, mem_path, device);
    let mut rng = StdRng::seed_from_u64(42);

    println!("ANT Chat (Rust). Type your message, empty line to quit.\n");

    let stdin = io::stdin();
    loop {
        print!("You: ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        stdin.lock().read_line(&mut input).unwrap();
        let input = input.trim();

        if input.is_empty() { break; }

        // Tokenize: tag + content
        let tagged = format!("localhost/user/chat@2025-01-01T00:00:00Z: {}\nlocalhost/ant/chat@2025-01-01T00:00:00Z: ", input);
        let mut prompt: Vec<u32> = vec![BOS_ID as u32];
        prompt.extend(tagged.bytes().map(|b| b as u32));

        let tokens = engine.generate(
            &prompt, 256, 0.8, 40, 0.9, &mut rng)?;

        // Decode, strip BOS/EOS/NOOP
        let bytes: Vec<u8> = tokens.iter()
            .filter(|&&t| t != BOS_ID as u32 && t != EOS_ID as u32 && t != NOOP_ID as u32)
            .map(|&t| t as u8)
            .collect();
        let response = detokenize(&bytes);

        println!("ANT: {}\n", response.trim());
    }

    engine.flush();
    Ok(())
}
