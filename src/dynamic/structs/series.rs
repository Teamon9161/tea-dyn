use super::DynTrustIter;
use tevec::polars::chunked_array::iterator::PolarsIterator;
use tevec::polars::prelude::Series;
use tevec::prelude::TResult;

pub trait SeriesExt {
    fn titer(&self) -> TResult<DynTrustIter>;
}

#[macro_export]
macro_rules! match_series {
    ($s: expr, $ca: ident; $body: expr) => {{
        use $crate::prelude::polars::prelude::DataType::*;
        match $s.dtype() {
            Boolean => {
                let $ca = $s.bool().unwrap();
                $body
            }
            Int32 => {
                let $ca = $s.i32().unwrap();
                $body
            }
            Int64 => {
                let $ca = $s.i64().unwrap();
                $body
            }
            Float32 => {
                let $ca = $s.f32().unwrap();
                $body
            }
            Float64 => {
                let $ca = $s.f64().unwrap();
                $body
            }
            _ => todo!(),
        }
    }};
}

impl SeriesExt for Series {
    fn titer(&self) -> TResult<DynTrustIter> {
        match_series!(self, ca; {
            let iter: Box<dyn PolarsIterator<Item = _> + '_> =
                    Box::new(ca.iter());
                Ok(iter.into())
        })
    }
}
