mod dynamic;
mod expr;

pub mod prelude;

#[cfg(feature = "ndarray")]
pub use tevec::ndarray;

// pub use dynamic::*;
// pub use expr::*;

// use pyo3::prelude::*;

// #[pymodule]
// fn _lowlevel(_py: Python, _m: &PyModule) -> PyResult<()> {
//     Ok(())
// }
