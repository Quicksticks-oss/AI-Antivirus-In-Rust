use ort::{
	tensor::OrtOwnedTensor, LoggingLevel, Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder, Value
};
use std::path::Path;
use ndarray::{Array, CowArray};
use ndarray::prelude::*;
use std::time::{Instant};

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

    let start_time = Instant::now();
    println!("Loading model...");

    let int_array: Array2<i32> = Array::from_shape_vec((1, 20), vec![1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]).expect("Failed to create int_array");
    let converted_array = int_array.map(|&x| x as i64);
    let cow_array: CowArray<i64, Ix2> = CowArray::from(converted_array);
    let farray = cow_array.clone().insert_axis(Axis(0)).into_shape((1, 20)).unwrap().into_dyn();
    let inputs = vec![Value::from_array(session.allocator(), &farray)?];
    println!("Running inference.");
    let outputs: Vec<Value> = session.run(inputs)?;

    let generated_tokens: OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
    let generated_tokens = generated_tokens.view();

    for row in generated_tokens.outer_iter() {
        for &value in row {
            print!("{} ", value);
        }
        println!();  // New line after each row
    }

    // Record the end time
    let end_time = Instant::now();

    // Calculate and print the elapsed time
    let elapsed_time = end_time - start_time;
    println!("Elapsed time: {:?}", elapsed_time);
    Ok(())
}
