mod ai {
    pub mod onnx;
}
mod tools {
    pub mod convert;
}
pub mod infer;

use ort::{
    Environment, ExecutionProvider, GraphOptimizationLevel, LoggingLevel, OrtResult, SessionBuilder,
};

use std::path::Path;
use std::time::Instant;

fn main() -> OrtResult<()> {
    let chunk_size = 1000000;
    let model_path: std::path::PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("MalwareModelSmall.onnx");
    println!("Starting AV.");

    match model_path.to_str() {
        Some(path_str) => println!("Path: {}", path_str),
        None => println!("Invalid Path"),
    }

    let environment = Environment::builder()
        .with_name("Encode")
        .with_log_level(LoggingLevel::Warning)
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?
        .with_model_from_file(model_path)
        .unwrap();

    println!("Loaded onnx model.");
    let start_time = Instant::now();

    println!("Running inference...");
    let file_path = "/media/reaktor/Data Drive/GithubRepos/AI-Antivirus-In-Rust/safe.dat";
    let is_virus = infer::infer_file(file_path, &session, chunk_size);
    println!("Is Virus: {}", is_virus);

    let end_time = Instant::now();
    // Calculate and print the elapsed time
    let elapsed_time = end_time - start_time;
    println!("Elapsed time: {:?}", elapsed_time);
    Ok(())
}
