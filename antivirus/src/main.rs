mod ai {
    pub mod onnx;
}

use ort::{
	LoggingLevel, Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder
};
use std::path::Path;
use ndarray::Array;
use ndarray::prelude::*;
use std::time::Instant;
use ai::onnx;

use memmap::MmapOptions;
use std::fs::File;

fn main() -> OrtResult<()> {
    let model_path: std::path::PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("model.onnx");
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

    let file_path = "/media/reaktor/Data Drive/GithubRepos/AI-Antivirus-In-Rust/safe.dat";

    let file = match File::open(file_path) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("Error opening file: {}", err);
            return Ok(());
        }
    };

    let mmap = match unsafe { MmapOptions::new().map(&file) } {
        Ok(mmap) => mmap,
        Err(err) => {
            eprintln!("Error creating memory map: {}", err);
            return Ok(());
        }
    };

    let data: &[u8] = &mmap;
    let num_elements = data.len();
    let shape = (1, num_elements);
    let array_data: Vec<i32> = data.iter().map(|&byte| byte as i32).collect();
    let int_array = Array2::from_shape_vec(Ix2(shape.0, shape.1), array_data).unwrap();

    let start_time = Instant::now();
    println!("Running inference...");

    //let int_array: Array2<i32> = Array::from_shape_vec((1, 10), vec![1,2,3,4,5,6,7,8,9,10]).expect("Failed to create int_array");
    //let int_array = array;
    let is_virus = onnx::run_onnx_inference(session, int_array);

    println!("Is Virus: {}", is_virus);

    // Record the end time
    let end_time = Instant::now();

    // Calculate and print the elapsed time
    let elapsed_time = end_time - start_time;
    println!("Elapsed time: {:?}", elapsed_time);
    Ok(())
}
