use super::py_any_ext::*;
use crate::expr::{Backend, Context, Data};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyList};
use std::borrow::Cow;
use std::collections::HashMap;

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
            if obj.module()?.name() != "pandas" {
                return Err(PyValueError::new_err("not a pandas DataFrame"));
            }
            let columns = obj.getattr("columns")?;
            let len = columns.len()?;
            let col_map = create_col_map_from_pyiter(&columns.iter()?, Some(len))?;
            let mut data = Vec::with_capacity(len);
            for col in columns.iter()? {
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

    #[cfg(feature = "pl")]
    fn from_polars(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        use pyo3_polars::PySeries;
        if obj.get_type().qualname()? == "DataFrame" {
            if obj.module()?.name() != "polars" {
                return Err(PyValueError::new_err("not a polars DataFrame"));
            }
            let series = obj.call_method0("get_columns")?;
            let len = obj.getattr("width")?.extract::<usize>()?;
            let mut data = Vec::with_capacity(len);
            let mut col_map = HashMap::with_capacity(len * 2);
            for (i, pyseries) in series.iter()?.enumerate() {
                let pyseries = pyseries?;
                let col = pyseries.getattr("name")?;
                let s = pyseries.extract::<PySeries>()?.0;
                data.push(Data::Series(s.into()));
                // safety: expression will only be evaluated when py context is alive
                let col = unsafe {
                    std::mem::transmute::<Cow<'_, str>, Cow<'py, str>>(
                        col.extract::<Cow<'_, str>>()?,
                    )
                };
                col_map.insert(col, i);
            }
            Ok(Context {
                data,
                backend: Some(Backend::Polars),
                col_map: Some(col_map),
            })
        } else {
            Err(PyValueError::new_err("not a polars DataFrame"))
        }
    }
}

impl<'py> FromPyObject<'py> for Context<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        #[cfg(feature = "pl")]
        if let Ok(ctx) = Context::from_polars(ob) {
            return Ok(ctx);
        }
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
