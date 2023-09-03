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
        .join("MalwareModelMedium.onnx");
    let model_path_str = model_path.to_str().unwrap();
    let chunk_size = 1048576;
    let session = ai::onnx::create_onnx_session(model_path_str).unwrap();

    println!("Loaded onnx model.");
    let start_time = Instant::now();

    let directory_path = "/"; // Change this to the directory you want to traverse.

    for entry in WalkDir::new(directory_path) {
        if let Ok(entry) = entry {
            let path = entry.path();
    
            if path.is_file() {
                if let Some(file_path) = path.to_str() {
                    if let Some(extension) = path.extension() {
                        if let Some(extension_str) = extension.to_str() {
                            let extension_str_lower = extension_str.to_lowercase();
                            if extension_str_lower == "exe" {
                                let is_virus = infer::infer_file(file_path, &session, chunk_size);
                                println!("Is Virus: {}, File: {:?}", is_virus, file_path);
                            }
                        }
                    }
                }
            }
            //println!("{:?}", path);
        } else {
            // Handle the error, e.g., print an error message or return an error code.
            eprintln!("Error: {:?}", entry.err());
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
