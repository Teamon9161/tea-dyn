use crate::prelude::*;
#[cfg(feature = "time")]
use numpy::datetime::{units, Datetime};
use numpy::PyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
    fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if obj.get_type().qualname()? == "Series" && obj.module()?.name() == "pandas" {
            Self::extract_bound(&obj.getattr("values")?)
        } else if let Ok(mut pyarray) = obj.extract::<PyArray<'py>>() {
            match_enum!(
                PyArray, &mut pyarray;
                (PureFloat | I32 | I64 | Bool | Object | Time)(arr)
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
                obj
            )))
        }
    }
}
