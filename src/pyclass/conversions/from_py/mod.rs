mod array;
mod context;
mod from_py_any;
mod py_any_ext;

pub use from_py_any::FromPyAny;
pub use py_any_ext::PyAnyExt;

use std::borrow::Cow;

use crate::prelude::*;

use pyo3::{exceptions::PyValueError, prelude::*};

#[cfg(feature = "time")]
use tevec::dtype::chrono::{DateTime as CrDateTime, Utc};

impl<'py> FromPyObject<'py> for Symbol {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(i) = ob.extract::<i32>() {
            Ok(Symbol::I32(i))
        } else if let Ok(i) = ob.extract::<usize>() {
            Ok(Symbol::Usize(i))
        } else if let Ok(s) = ob.extract::<Cow<'_, str>>() {
            match s {
                Cow::Owned(s) => Ok(s.into()),
                Cow::Borrowed(s) => Ok(s.to_owned().into()),
            }
        } else {
            Err(PyValueError::new_err(format!(
                "can not convert {:?} to Symbol",
                ob
            )))
        }
    }
}

impl<'py> FromPyObject<'py> for Backend {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = obj.extract::<Cow<'_, str>>() {
            match s.as_ref() {
                "pandas" | "pd" => Ok(Backend::Pandas),
                "numpy" | "np" => Ok(Backend::Numpy),
                "vec" | "list" => Ok(Backend::Vec),
                // "polars" | "pl" => Ok(Backend::Polars),
                _ => Err(PyValueError::new_err(format!(
                    "can not convert {:?} to Backend",
                    s
                ))),
            }
        } else {
            Err(PyValueError::new_err(format!(
                "can not convert {:?} to Backend",
                obj
            )))
        }
    }
}

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
                    let v: DateTime = v.into();
                    return Ok(v.into());
                }
                // TODO: ability to extract timedelta from python sid
            }
            Ok(Object(obj.to_object(obj.py())).into())
        }
    }
}

impl<'py> FromPyObject<'py> for Data<'py> {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(arr) = obj.extract::<DynArray>() {
            Ok(arr.into())
        } else if let Ok(vec) = obj.extract::<DynVec>() {
            Ok(vec.into())
        } else if let Ok(scalar) = obj.extract::<Scalar>() {
            Ok(scalar.into())
        } else {
            // this should be unreachable as extract to a scalar should never fail
            unreachable!()
        }
    }
}
