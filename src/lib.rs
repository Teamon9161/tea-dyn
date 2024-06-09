mod dynamic;
mod expr;
#[cfg(feature = "py")]
mod py_dtype;
#[cfg(feature = "py")]
mod pyclass;
#[cfg(feature = "py")]
use pyo3::prelude::*;

pub mod prelude;

#[cfg(feature = "ndarray")]
pub use tevec::ndarray;

#[cfg(feature = "py")]
#[pymodule]
fn _lowlevel(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    use pyclass::*;
    m.add_class::<PyExpr>()?;
    m.add_function(wrap_pyfunction!(py_lit, m)?)?;
    m.add_function(wrap_pyfunction!(py_s, m)?)?;
    Ok(())
}
