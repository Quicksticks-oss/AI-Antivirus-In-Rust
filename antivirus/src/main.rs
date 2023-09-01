mod ai {
    pub mod onnx;
}
mod tools {
    pub mod convert;
}
pub mod infer;
use ort::OrtResult;
use std::path::Path;
use std::time::Instant;

fn main() -> OrtResult<()> {
    let chunk_size = 1000000;
    let model_path: std::path::PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("MalwareModelSmall.onnx");
    println!("Starting AV.");

    let model_path_str = model_path.to_str().unwrap();
    let session = ai::onnx::create_onnx_session(model_path_str).unwrap();

    println!("Loaded onnx model.");
    let start_time = Instant::now();

    println!("Running inference...");
    let file_path = "/media/reaktor/Data Drive/GithubRepos/AI-Antivirus-In-Rust/test.dat";
    let is_virus = infer::infer_file(file_path, &session, chunk_size);
    println!("Is Virus: {}", is_virus);

    let end_time = Instant::now();
    // Calculate and print the elapsed time
    let elapsed_time = end_time - start_time;
    println!("Elapsed time: {:?}", elapsed_time);
    Ok(())
}
