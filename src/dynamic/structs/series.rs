use super::DynTrustIter;
use tevec::polars::chunked_array::iterator::PolarsIterator;
use tevec::polars::prelude::{DataType as PlDataType, Series};
use tevec::prelude::TResult;

pub trait SeriesExt {
    fn titer(&self) -> TResult<DynTrustIter>;
}

impl SeriesExt for Series {
    fn titer(&self) -> TResult<DynTrustIter> {
        use PlDataType::*;
        match self.dtype() {
            Boolean => {
                let iter: Box<dyn PolarsIterator<Item = _> + '_> =
                    Box::new(self.bool().unwrap().iter());
                Ok(iter.into())
            }
            Int32 => {
                let iter: Box<dyn PolarsIterator<Item = _> + '_> =
                    Box::new(self.i32().unwrap().iter());
                Ok(iter.into())
            }
            Int64 => {
                let iter: Box<dyn PolarsIterator<Item = _> + '_> =
                    Box::new(self.i64().unwrap().iter());
                Ok(iter.into())
            }
            Float32 => {
                let iter: Box<dyn PolarsIterator<Item = _> + '_> =
                    Box::new(self.f32().unwrap().iter());
                Ok(iter.into())
            }
            Float64 => {
                let iter: Box<dyn PolarsIterator<Item = _> + '_> =
                    Box::new(self.f64().unwrap().iter());
                Ok(iter.into())
            }
            _ => todo!(),
        }
    }
}
