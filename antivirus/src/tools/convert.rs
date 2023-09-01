use ndarray::{Array, Axis, Ix2};

pub fn split_tensor(input_tensor: Array<i32, Ix2>, chunk_size: usize) -> Vec<Array<i32, Ix2>> {
    if input_tensor.shape()[1] > chunk_size {
        return input_tensor.axis_chunks_iter(Axis(1), chunk_size)
            .map(|chunk| chunk.to_owned())
            .collect();
    } else {
        return vec![input_tensor];
    }
}