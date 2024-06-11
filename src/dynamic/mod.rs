mod methods;
mod structs;
#[macro_use]
mod macros;

pub use structs::*;

#[cfg(feature = "py")]
use crate::py_dtype::Object;
#[cfg(feature = "time")]
use tevec::prelude::{unit, DateTime, TimeDelta};
use tevec::{dtype::GetDataType, prelude::DataType};

/// A trait to get the actual datatype from an enum
/// different from GetDataType, this function need a reference to self as input
pub trait GetDtype {
    fn dtype(&self) -> DataType;
}

/// A wrapper trait for GetDataType of tevec
/// # Safety
///
/// Any type implement this trait must ensure that the type has the same size as
/// the type which DataType represents
pub unsafe trait Dtype {
    fn type_() -> DataType;
}

macro_rules! impl_dtype {
    ($($ty: ty),*) => {
        $(
            unsafe impl Dtype for $ty{
                #[inline]
                fn type_() -> DataType {
                    <$ty>::dtype()
                }
            }
        )*
    };
}

impl_dtype!(
    bool,
    f32,
    f64,
    i32,
    i64,
    u8,
    u64,
    usize,
    String,
    Option<usize>,
    Vec<usize>
);

#[cfg(feature = "time")]
impl_dtype!(
    DateTime<unit::Millisecond>,
    DateTime<unit::Microsecond>,
    DateTime<unit::Nanosecond>,
    TimeDelta
);

#[cfg(feature = "py")]
impl_dtype!(Object);
