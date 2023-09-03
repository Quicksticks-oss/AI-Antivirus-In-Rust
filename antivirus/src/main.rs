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
use walkdir::WalkDir;

fn main() -> OrtResult<()> {
    let model_path: std::path::PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("MalwareModelNew.onnx");
    let model_path_str = model_path.to_str().unwrap();
    let chunk_size = 1048576;
    let session = ai::onnx::create_onnx_session(model_path_str).unwrap();

    println!("Loaded onnx model.");
    let start_time = Instant::now();

    let directory_path = "/media/reaktor/Data Drive/GithubRepos/AI-Antivirus-In-Rust/test"; // Change this to the directory you want to traverse.

    for entry in WalkDir::new(directory_path) {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.is_file() {
            let file_path = path.to_str().unwrap_or("");
            let extension_str: String = path.extension().unwrap().to_str().unwrap_or("").to_lowercase();
            if extension_str == "exe"{
                let is_virus = infer::infer_file(file_path, &session, chunk_size);
                println!("Is Virus: {}, File: {:?}", is_virus, file_path);
            }
        }
    }

    //println!("Running inference...");
    //let file_path = "/media/reaktor/Data Drive/GithubRepos/AI-Antivirus-In-Rust/test.dat";
    //let is_virus = infer::infer_file(file_path, &session, chunk_size);
    //println!("Is Virus: {}", is_virus);

    let end_time = Instant::now();
    // Calculate and print the elapsed time
    let elapsed_time = end_time - start_time;
    println!("Elapsed time: {:?}", elapsed_time);
    Ok(())
}
