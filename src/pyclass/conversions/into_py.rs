use crate::prelude::*;
use numpy::{Element, PyArray};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyList};
#[cfg(feature = "pl")]
use pyo3_polars::PySeries;

#[cfg(feature = "time")]
use numpy::datetime::{units, Datetime as NPDatetime};

impl<'a> IntoPy<PyObject> for DynVec<'a> {
    #[allow(unreachable_patterns)]
    fn into_py(self, py: Python) -> PyObject {
        match_vec!(self;
            (Normal | String | VecUsize)(e) => Ok(e.into_owned().into_py(py)),
            Time(vec) => Ok(vec.iter().map(|v| v.to_cr().unwrap()).collect_trusted_to_vec().into_py(py)),
            Object(vec) => Ok(unsafe{TransmuteDtype::<PyObject>::into_dtype(vec.into_owned())}.into_py(py)),
        )
        .unwrap()
    }
}

impl IntoPy<PyObject> for Scalar {
    #[allow(unreachable_patterns)]
    #[inline]
    fn into_py(self, py: Python) -> PyObject {
        // TODO: support TimeDelta
        match_scalar!(self;
            (Normal | String | VecUsize)(s) => Ok(s.into_py(py)),
            Time(dt) => Ok(dt.to_cr().unwrap().into_py(py)),
            Object(obj) => Ok(obj.0), // TODO: this will panic if reference from python and return to python directly
        )
        .unwrap()
    }
}

impl<'a, T: Element> ArbArray<'a, T> {
    #[inline]
    fn try_into_py(self, py: Python, container: Option<Bound<'_, PyAny>>) -> PyResult<PyObject> {
        match self {
            ArbArray::Owned(arr) => Ok(PyArray::from_owned_array_bound(py, arr).into_py(py)),
            ArbArray::View(arr) => {
                // TODO: Maybe we should clone the array if the context is not passed?
                // or we just keep raising an error?
                // I think this case will only happen when the expression is just select a
                // column from the context without any operation, so there should be a context
                // passed to the function
                let container = container.ok_or_else(|| {
                    PyValueError::new_err(
                        "Should pass a context so that ArrayView can be borrowed by numpy array",
                    )
                })?;
                let arr = unsafe { PyArray::borrow_from_array_bound(&arr, container) };
                Ok(arr.into_py(py))
            }
            ArbArray::ViewMut(arr) => {
                // TODO: can we optimize this case? probably not
                let container = container.ok_or_else(|| {
                    PyValueError::new_err(
                        "Should pass a context so that ArrayView can be borrowed by numpy array",
                    )
                })?;
                let arr = unsafe { PyArray::borrow_from_array_bound(&arr, container) };
                Ok(arr.into_py(py))
            }
        }
    }
}

impl<'a> DynArray<'a> {
    pub fn try_into_py(
        self,
        py: Python,
        container: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        // TODO: support more types, such as TimeDelta, String, Option<usize> and so on
        match_array!(self;
            PureNumeric(arr) | Object(arr) => Ok(arr.try_into_py(py, container)?),
            #[cfg(feature = "time")]
            DateTimeMs(arr) => {
                // safety: Numpy Datetime has the same memory layout with tevec DateTime
                let np_arr: ArbArray<'_, NPDatetime<units::Milliseconds>> = unsafe{std::mem::transmute(arr)};
                Ok(np_arr.try_into_py(py, container)?)
            },
            #[cfg(feature = "time")]
            DateTimeUs(arr) => {
                // safety: Numpy Datetime has the same memory layout with tevec DateTime
                let np_arr: ArbArray<'_, NPDatetime<units::Microseconds>> = unsafe{std::mem::transmute(arr)};
                Ok(np_arr.try_into_py(py, container)?)
            },
            #[cfg(feature = "time")]
            DateTimeNs(arr) => {
                // safety: Numpy Datetime has the same memory layout with tevec DateTime
                let np_arr: ArbArray<'_, NPDatetime<units::Nanoseconds>> = unsafe{std::mem::transmute(arr)};
                Ok(np_arr.try_into_py(py, container)?)
            },
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))
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
            Data::Vec(vec) => {
                // python list have a different memory layout with Vec
                // so a clone is always needed
                // TODO: TimeDelta is not supported yet
                match_vec!(vec.as_ref();
                    AsRefPy(vec) => Ok(PyList::new_bound(py, vec.iter()).into_py(py)),
                    Object(vec) => Ok(PyList::new_bound(py, vec.iter().cloned()).into_py(py)),
                    Time(vec) => Ok(PyList::new_bound(py, vec.iter().map(|v| v.to_cr().unwrap())).into_py(py)),
                ).map_err(|e| PyValueError::new_err(e.to_string()))
            }
            Data::Array(arr) => match Arc::try_unwrap(arr) {
                Ok(arr) => arr.try_into_py(py, container),
                Err(arr) => {
                    // array is shared, bind it to the container
                    arr.view().try_into_py(py, container)
                }
            },
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
            #[cfg(feature = "pl")]
            Data::Series(series) => Ok(PySeries(series).into_py(py)),
        }
    }
}
