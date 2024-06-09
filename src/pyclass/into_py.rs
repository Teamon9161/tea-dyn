use crate::prelude::*;
use pyo3::prelude::*;

impl IntoPy<PyObject> for DynVec {
    fn into_py(self, py: Python) -> PyObject {
        match_vec!(self;
            normal(vec) => Ok(vec.into_py(py)),
            Object(vec) => Ok(unsafe{TransmuteDtype::<PyObject>::into_dtype(vec)}.into_py(py)),
        )
        .unwrap()
    }
}

impl<'py> IntoPy<PyObject> for Data<'py> {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            // Data::TrustIter(iter) => iter.to_object(py),
            // Data::Scalar(scalar) => scalar.to_object(py),
            Data::Vec(vec) => Arc::try_unwrap(vec).unwrap().into_py(py),
            // Data::Array(arr) => arr.to_object(py),
            _ => todo!(),
        }
    }
}
