use ndarray::{Array2, Axis, CowArray, Ix2};
use ort::{Session, Value};
use ort::tensor::ort_owned_tensor::ViewHolder;

use ort::{
    Environment, ExecutionProvider, GraphOptimizationLevel, LoggingLevel, OrtError, SessionBuilder,
};


pub fn create_onnx_session(model_path: &str) -> Result<Session, OrtError>{
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

    Ok(session)
}

pub fn run_onnx_inference(ort_ession: &Session, int_array: &Array2<i32>) -> bool {
    //let shape = int_array.shape();

    // Remove all 0s from the array
    let non_zero_elements = int_array.iter().filter(|&&x| x != 0).cloned().collect::<Vec<_>>();


    // Calculate the dimensions of the new array
    let shape = (1, non_zero_elements.len() as usize);

    // Create a new 2D array without 0s
    let int_array = Array2::from_shape_vec(shape, non_zero_elements).unwrap();


    let converted_array = int_array.map(|&x| x as i32);
    let cow_array: CowArray<i32, Ix2> = CowArray::from(converted_array);

    
    let farray = cow_array
        .clone()
        .insert_axis(Axis(0))
        .into_shape(shape)
        .unwrap()
        .into_dyn();

    let results = vec![Value::from_array(ort_ession.allocator(), &farray)];
    let values: Vec<Value<'_>> = results
        .into_iter() // Convert the vector into an iterator
        .filter_map(|result| result.ok())
        .collect();
    

    let outputs: Vec<Value<'_>> = ort_ession.run(values).unwrap();
    
    let generated_tokens = outputs[0].try_extract().expect("Token error.");
    
    let generated_tokens: ViewHolder<'_, f32, _> = generated_tokens.view();
    //let generated_tokens_view: ArrayBase<ndarray::ViewRepr<&f32>, Dim<IxDynImpl>> = generated_tokens.view();

    //println!("{}", generated_tokens_view);

    let mut min_value = f32::NEG_INFINITY;
    let mut min_index = 0;

    for (index, &value) in generated_tokens.iter().enumerate() {
        if value > min_value {
            min_value = value;
            min_index = index;
        }
    }

    //println!("Max value: {}", min_value);
    //println!("Index of max value: {}", min_index);
    //println!("{:?}", generated_tokens.view());

    let result = min_index != generated_tokens.shape()[1]-1;

    result
}
