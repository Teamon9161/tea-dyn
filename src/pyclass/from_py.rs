use crate::prelude::*;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};
use tevec::dtype::chrono::{DateTime as CrDateTime, Utc};

impl<'py> FromPyObject<'py> for DynVec {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        let vec = if let Ok(v) = obj.extract::<Vec<bool>>() {
            v.into()
        } else if let Ok(v) = obj.extract::<Vec<i32>>() {
            v.into()
        } else if let Ok(v) = obj.extract::<Vec<f64>>() {
            v.into()
        } else if let Ok(v) = obj.extract::<Vec<Option<usize>>>() {
            v.into()
        } else if let Ok(v) = obj.extract::<Vec<String>>() {
            v.into()
        } else if let Ok(v) = obj.extract::<Vec<Object>>() {
            v.into()
        } else {
            return Err(PyValueError::new_err(format!(
                "can not cast python object {obj:?} to vec"
            )));
        };
        Ok(vec)
    }
}

impl<'a> FromPyObject<'a> for Scalar {
    fn extract_bound(obj: &Bound<'a, PyAny>) -> PyResult<Self> {
        if let Ok(v) = obj.extract::<bool>() {
            Ok(v.into())
        } else if let Ok(v) = obj.extract::<i32>() {
            Ok(v.into())
        } else if let Ok(v) = obj.extract::<f64>() {
            Ok(v.into())
        } else if let Ok(v) = obj.extract::<Option<usize>>() {
            Ok(v.into())
        } else if let Ok(v) = obj.extract::<String>() {
            Ok(v.into())
        } else {
            #[cfg(feature = "time")]
            {
                if let Ok(v) = obj.extract::<CrDateTime<Utc>>() {
                    return Ok(DateTime(Some(v)).into());
                }
                // TODO: ability to extract timedelta from python sid
            }
            Ok(Object(obj.to_object(obj.py())).into())
        }
    }
}

impl<'py> FromPyObject<'py> for Data<'py> {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(vec) = obj.extract::<DynVec>() {
            Ok(vec.into())
        } else {
            todo!()
        }
    }
}

impl<'py> FromPyObject<'py> for Context<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(_pydict) = ob.downcast::<PyDict>() {
            todo!()
        } else if let Ok(pylist) = ob.downcast::<PyList>() {
            let ctx: Vec<Data<'py>> = pylist.into_iter().map(|v| v.extract().unwrap()).collect();
            Ok(Context { data: ctx })
        } else {
            todo!()
        }
    }
}
