use std::borrow::BorrowMut;

use crate::prelude::*;
#[cfg(feature = "time")]
use numpy::datetime::{units, Datetime};
use numpy::PyArrayDyn;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
};
#[cfg(feature = "time")]
use tevec::dtype::chrono::{DateTime as CrDateTime, Utc};

#[derive(FromPyObject)]
pub enum PyArray<'py> {
    Bool(&'py PyArrayDyn<bool>),
    F32(&'py PyArrayDyn<f32>),
    F64(&'py PyArrayDyn<f64>),
    I32(&'py PyArrayDyn<i32>),
    I64(&'py PyArrayDyn<i64>),
    Object(&'py PyArrayDyn<Object>),
    #[cfg(feature = "time")]
    DateTimeMs(&'py PyArrayDyn<Datetime<units::Milliseconds>>),
    #[cfg(feature = "time")]
    DateTimeUs(&'py PyArrayDyn<Datetime<units::Microseconds>>),
    #[cfg(feature = "time")]
    DateTimeNs(&'py PyArrayDyn<Datetime<units::Nanoseconds>>),
}

impl<'py> FromPyObject<'py> for DynArray<'py> {
    #[allow(unreachable_patterns)]
    fn extract_bound(mut ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(mut pyarray) = ob.borrow_mut().extract::<PyArray<'py>>() {
            match_enum!(
                PyArray, &mut pyarray;
                // Bool(arr) | F32(arr) | F64(arr) | I32(arr) | I64(arr) | Object(arr)
                // | #[cfg(feature = "time")] DateTimeMs(arr)
                // | #[cfg(feature = "time")] DateTimeUs(arr)
                // | #[cfg(feature = "time")] DateTimeNs(arr)
                (Float | I32 | I64 | Bool | Object | Time)(arr)
                => {
                    if let Ok(mut arr) = arr.try_readwrite() {
                        let arb_arr: ArbArray<'_, _> = arr.as_array_mut().into();
                        // safety: this is only safe when python side keeps the array alive
                        // since expressions will only be evaluated when context is alive,
                        // we can safely assume that the array is alive
                        let arb_arr: ArbArray<'py, _> = unsafe { arb_arr.into_life()};
                        Ok(arb_arr.into())
                    } else {
                        let arr = arr.try_readonly()?;
                        let arb_arr: ArbArray<'_, _> = arr.as_array().into();
                        // safety: this is only safe when python side keeps the array alive
                        // since expressions will only be evaluated when context is alive,
                        // we can safely assume that the array is alive
                        let arb_arr: ArbArray<'py, _> = unsafe { arb_arr.into_life()};
                        Ok(arb_arr.into())
                    }
                },
                // Object(arr) =>
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
        } else {
            Err(PyValueError::new_err(format!(
                "can not convert {:?} to DynArray",
                ob
            )))
        }
    }
}

impl<'py> FromPyObject<'py> for Backend {
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(s) = obj.extract::<&str>() {
            match s {
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

impl<'py> FromPyObject<'py> for Context<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(_pydict) = ob.downcast::<PyDict>() {
            todo!()
        } else if let Ok(pylist) = ob.downcast::<PyList>() {
            let ctx: Vec<Data<'py>> = pylist.into_iter().map(|v| v.extract().unwrap()).collect();
            Ok(Context {
                data: ctx,
                backend: None,
            })
        } else {
            todo!()
        }
    }
}
