#[cfg(feature = "ndarray")]
mod array;
mod scalar;
#[cfg(feature = "pl")]
mod series;
mod trust_iter;
mod vec;

#[cfg(feature = "ndarray")]
pub use array::{ArbArray, DynArray, NdArrayExt};
pub use scalar::Scalar;
#[cfg(feature = "pl")]
pub use series::SeriesExt;
pub use trust_iter::{DynTrustIter, TrustIterCast, TvIter};
pub use vec::DynVec;

pub trait TransmuteDtype<T> {
    type Output;
    /// # Safety
    ///
    /// the caller must ensure T and Self is actually the same type
    unsafe fn into_dtype(self) -> Self::Output;
}
