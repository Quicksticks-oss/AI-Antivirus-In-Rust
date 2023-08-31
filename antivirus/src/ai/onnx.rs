use ndarray::{Array2, Axis, CowArray, Ix2};
use ort::{Session, Value};
use ort::tensor::ort_owned_tensor::ViewHolder;

pub fn run_onnx_inference(ort_ession: Session, int_array: Array2<i32>) -> bool {
    let shape = int_array.shape();
    println!("{:?}", shape);
    let converted_array = int_array.map(|&x| x as i64);
    let cow_array: CowArray<i64, Ix2> = CowArray::from(converted_array);
    let farray = cow_array
        .clone()
        .insert_axis(Axis(0))
        .into_shape(shape)
        .unwrap()
        .into_dyn();

    let results = vec![Value::from_array(ort_ession.allocator(), &farray)];
    let values = results
        .into_iter() // Convert the vector into an iterator
        .filter_map(|result| result.ok())
        .collect();

    let outputs: Vec<Value<'_>> = ort_ession.run(values).unwrap();
    
    let generated_tokens = outputs[0].try_extract().expect("Token error.");
    
    let generated_tokens: ViewHolder<'_, f32, _> = generated_tokens.view();
    let generated_tokens_view = generated_tokens.view();

    let value_00 = generated_tokens_view[[0, 0]];
    let value_01 = generated_tokens_view[[0, 1]];

    let result = value_00 > value_01;

    result
}
