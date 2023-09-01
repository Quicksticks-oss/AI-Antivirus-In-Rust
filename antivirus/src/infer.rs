use crate::ai::onnx;
use crate::tools::convert;
use memmap::MmapOptions;
use ndarray::prelude::*;
use ort::Session;
use std::fs::File;

pub fn infer_file(file_path: &str, ort_session: &Session, chunk_size: usize) -> bool {
    let file = match File::open(file_path) {
        Ok(file) => file,
        Err(err) => {
            eprintln!("Error opening file: {}", err);
            return false;
        }
    };

    let mmap = match unsafe { MmapOptions::new().map(&file) } {
        Ok(mmap) => mmap,
        Err(err) => {
            eprintln!("Error creating memory map: {}", err);
            return false;
        }
    };

    let data: &[u8] = &mmap;
    let num_elements = data.len();
    let shape = (1, num_elements);
    let array_data: Vec<i32> = data.iter().map(|&byte| byte as i32).collect();
    let int_array = Array2::from_shape_vec(Ix2(shape.0, shape.1), array_data).unwrap();

    let chunks = convert::split_tensor(int_array, chunk_size);

    let is_virus = false;

    for chunk in &chunks {
        let is_virus = onnx::run_onnx_inference(ort_session, chunk);
        if is_virus == true {
            return is_virus;
        }
    }

    return is_virus;
}
