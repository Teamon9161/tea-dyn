use super::DynTrustIter;
use tevec::polars::chunked_array::iterator::PolarsIterator;
use tevec::polars::prelude::{DataType as PlDataType, Series};
use tevec::prelude::TResult;

pub trait SeriesExt {
    fn to_titer(&self) -> TResult<DynTrustIter>;
}

impl SeriesExt for Series {
    fn to_titer(&self) -> TResult<DynTrustIter> {
        use PlDataType::*;
        match self.dtype() {
            Int32 => {
                let iter: Box<dyn PolarsIterator<Item = _> + '_> =
                    Box::new(self.i32().unwrap().iter());
                Ok(iter.into())
            }
            // Int64 => Ok(s.i64().unwrap().into_iter()),
            // Float32 => Ok(s.f32().unwrap().into_iter()),
            _ => todo!(),
        }
    }
}
