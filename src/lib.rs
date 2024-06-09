mod dtype;
mod dynamic;
mod expr;

pub mod prelude;

#[cfg(feature = "ndarray")]
pub use tevec::ndarray;

// #[pymodule]
// fn _lowlevel(_py: Python, _m: &PyModule) -> PyResult<()> {
//     Ok(())
// }
