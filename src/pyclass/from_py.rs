use std::{
    borrow::{BorrowMut, Cow},
    collections::HashMap,
};

use crate::prelude::*;
#[cfg(feature = "time")]
use numpy::datetime::{units, Datetime};
use numpy::PyArrayDyn;
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyIterator, PyList},
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

impl<'py> FromPyObject<'py> for Symbol {
    fn extract(ob: &PyAny) -> PyResult<Self> {
        if let Ok(i) = ob.extract::<i32>() {
            Ok(Symbol::I32(i))
        } else if let Ok(i) = ob.extract::<usize>() {
            Ok(Symbol::Usize(i))
        } else if let Ok(s) = ob.extract::<Cow<'_, str>>() {
            match s {
                Cow::Owned(s) => Ok(Symbol::String(s)),
                Cow::Borrowed(s) => Ok(Symbol::String(s.to_owned())),
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

impl<'py> FromPyObject<'py> for Context<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        // Ok(Context::from_pandas(ob).unwrap())
        if let Ok(ctx) = Context::from_pandas(ob) {
            Ok(ctx)
        } else if let Ok(ctx) = Context::from_dict(ob) {
            Ok(ctx)
        } else if let Ok(pylist) = ob.downcast::<PyList>() {
            let ctx: Vec<Data<'py>> = pylist.into_iter().map(|v| v.extract().unwrap()).collect();
            Ok(Context {
                data: ctx,
                backend: None,
                col_map: None,
            })
        } else {
            Err(PyValueError::new_err(format!(
                "{:?} is not a valid context",
                ob
            )))
        }
    }
}

fn create_col_map_from_pyiter<'py>(
    columns: &Bound<'py, PyIterator>,
    len: Option<usize>,
) -> PyResult<HashMap<Cow<'py, str>, usize>> {
    let len = len.unwrap_or_else(|| columns.len().unwrap());
    let mut col_map = HashMap::with_capacity(len * 2);
    for (i, col) in columns.into_iter().enumerate() {
        let col = col?;
        let col = col.extract::<Cow<'_, str>>()?;
        // safety: expression will only be evaluated when py context is alive
        let col = unsafe { std::mem::transmute::<Cow<'_, str>, Cow<'py, str>>(col) };
        col_map.insert(col, i);
    }
    Ok(col_map)
}

fn create_col_map_from_pylist<'py>(
    columns: &Bound<'py, PyList>,
) -> PyResult<HashMap<Cow<'py, str>, usize>> {
    let mut col_map = HashMap::with_capacity(columns.len() * 2);
    for (i, col) in columns.into_iter().enumerate() {
        let col = col.extract::<Cow<'_, str>>()?;
        // safety: expression will only be evaluated when py context is alive
        let col = unsafe { std::mem::transmute::<Cow<'_, str>, Cow<'py, str>>(col) };
        col_map.insert(col, i);
    }
    Ok(col_map)
}

impl<'py> Context<'py> {
    fn from_dict(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(pydict) = obj.downcast::<PyDict>() {
            let keys = pydict.keys();
            let col_map = create_col_map_from_pylist(&keys)?;
            let mut data = Vec::with_capacity(keys.len());
            for col in keys {
                let value = pydict.get_item(col)?.unwrap();
                data.push(value.extract::<Data<'py>>()?)
            }
            Ok(Context {
                data,
                backend: None,
                col_map: Some(col_map),
            })
        } else {
            Err(PyValueError::new_err("not a dict"))
        }
    }

    fn from_pandas(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if obj.get_type().qualname()? == "DataFrame" {
            let module_name = obj.getattr("__module__")?;
            if module_name
                .extract::<Cow<'_, str>>()?
                .split('.')
                .next()
                .unwrap()
                != "pandas"
            {
                return Err(PyValueError::new_err("not a pandas DataFrame"));
            }
            let columns = obj.getattr("columns")?;
            let len = columns.len()?;
            let col_map =
                create_col_map_from_pyiter(&PyIterator::from_bound_object(&columns)?, Some(len))?;
            let mut data = Vec::with_capacity(len);
            for col in PyIterator::from_bound_object(&columns)? {
                let value = obj.get_item(col?)?.getattr("values")?;
                data.push(value.extract::<Data<'py>>()?)
            }
            Ok(Context {
                data,
                backend: Some(Backend::Pandas),
                col_map: Some(col_map),
            })
        } else {
            Err(PyValueError::new_err("not a pandas DataFrame"))
        }
    }
}
