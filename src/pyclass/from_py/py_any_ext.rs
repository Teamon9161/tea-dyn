use pyo3::types::PyString;
use pyo3::{intern, prelude::*};

#[repr(transparent)]
pub struct PyModuleName<'py>(Bound<'py, PyAny>);

impl<'py> PyModuleName<'py> {
    pub fn name(&self) -> &str {
        self.0
            .downcast::<PyString>()
            .unwrap()
            .as_gil_ref()
            .to_str()
            .unwrap()
            .split('.')
            .next()
            .unwrap()
    }
}

pub trait PyAnyExt {
    fn module(&self) -> PyResult<PyModuleName>;
    fn as_str(&self) -> PyResult<&str>;
}

impl<'py> PyAnyExt for Bound<'py, PyAny> {
    #[inline]
    fn module(&self) -> PyResult<PyModuleName> {
        self.getattr(intern!(self.py(), "__module__"))
            .map(PyModuleName)
    }

    #[inline]
    fn as_str(&self) -> PyResult<&str> {
        self.downcast::<PyString>()?.as_gil_ref().to_str()
    }
}
