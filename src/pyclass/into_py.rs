use crate::prelude::*;
use numpy::PyArray;
use pyo3::{exceptions::PyValueError, prelude::*};

impl IntoPy<PyObject> for DynVec {
    #[allow(unreachable_patterns)]
    fn into_py(self, py: Python) -> PyObject {
        match_vec!(self;
            (Normal | String | VecUsize)(e) => Ok(e.into_py(py)),
            Time(vec) => Ok(vec.into_iter().map(|v| v.to_cr().unwrap()).collect_trusted_to_vec().into_py(py)),
            Object(vec) => Ok(unsafe{TransmuteDtype::<PyObject>::into_dtype(vec)}.into_py(py)),
        )
        .unwrap()
    }
}

impl IntoPy<PyObject> for Scalar {
    #[allow(unreachable_patterns)]
    #[inline]
    fn into_py(self, py: Python) -> PyObject {
        match_scalar!(self;
            (Normal | String | VecUsize)(vec) => Ok(vec.into_py(py)),
            Time(dt) => Ok(dt.to_cr().unwrap().into_py(py)),
            Object(obj) => Ok(obj.0),
        )
        .unwrap()
    }
}

impl<'a> DynArray<'a> {
    pub fn try_into_py(
        self,
        py: Python,
        container: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        match_array!(self;
            PureNumeric(arr) => {
                match arr {
                    ArbArray::Owned(arr) => Ok(PyArray::from_owned_array_bound(py, arr).into_py(py)),
                    ArbArray::View(arr) => {
                        // TODO: Maybe we should clone the array if the context is not passed?
                        // or we just keep raising an error?
                        // I think this case will only happen when the expression is just select a
                        // column from the context without any operation, so there should be a context
                        // passed to the function
                        let container = container.ok_or_else(|| PyValueError::new_err("Should pass a context so that ArrayView can be borrowed by numpy array"))?;
                        let arr = unsafe {PyArray::borrow_from_array_bound(&arr, container)};
                        Ok(arr.into_py(py))
                    }
                    ArbArray::ViewMut(arr) => {
                        // TODO: can we optimize this case? probably not
                        let container = container.ok_or_else(|| PyValueError::new_err("Should pass a context so that ArrayView can be borrowed by numpy array"))?;
                        let arr = unsafe {PyArray::borrow_from_array_bound(&arr, container)};
                        Ok(arr.into_py(py))
                    }
                }
                // todo!()
            },
        ).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

impl<'py> Data<'py> {
    pub fn try_into_py(
        self,
        py: Python,
        container: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        match self {
            Data::Scalar(scalar) => match Arc::try_unwrap(scalar) {
                Ok(scalar) => Ok(scalar.into_py(py)),
                // TODO: deal with types that can not be cheap cloned
                Err(scalar) => Ok(scalar.cheap_clone().unwrap().into_py(py)),
            },
            Data::Vec(vec) => Ok(Arc::try_unwrap(vec).unwrap().into_py(py)),
            Data::Array(arr) => Arc::try_unwrap(arr)
                .expect("array is shared")
                .try_into_py(py, container),
            Data::TrustIter(iter) => {
                let iter = Arc::try_unwrap(iter).map_err(|_| {
                    PyValueError::new_err("iter is shared, can not cast to python type")
                })?;
                let vec = iter
                    .collect_vec()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let array = DynArray::from_vec(vec).unwrap();
                array.try_into_py(py, container)
            }
        }
    }
}
