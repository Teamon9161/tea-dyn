use pyo3::prelude::*;

pub trait FromPyAny<'py>
where
    Self: Sized,
{
    fn from_py_any(obj: &Bound<'py, PyAny>) -> PyResult<Self>;
}
