use ort::{
	tensor::OrtOwnedTensor, LoggingLevel, Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder, Value
};
use std::path::Path;
use ndarray::{Array, CowArray};
use ndarray::prelude::*;

fn main() -> OrtResult<()> {
    println!("Loading model...");
    let model_path: std::path::PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("model.onnx");
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

    let int_array: Array<i32, _> = Array::from_shape_vec((1, 10), vec![1, 2, 3, 4, 5, 6,7,8,9,10]).unwrap();
    let float_array: Array2<f32> = int_array.mapv(|x| x as f32);
    let numpy_array = CowArray::from(Array::from(float_array));
    let farray = numpy_array.clone().insert_axis(Axis(0)).into_shape((1, 10)).unwrap().into_dyn();

    let inputs = vec![Value::from_array(session.allocator(), &farray)?];
    let outputs: Vec<Value> = session.run(inputs)?;
    let generated_tokens: OrtOwnedTensor<f32, _> = outputs[0].try_extract()?;
    let generated_tokens = generated_tokens.view();

    for row in generated_tokens.outer_iter() {
        for &value in row {
            print!("{} ", value);
        }
        println!();  // New line after each row
    }

    Ok(())
}
