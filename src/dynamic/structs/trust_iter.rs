use crate::prelude::*;
use tea_macros::GetDtype;

impl<'a, T, U> TransmuteDtype<U> for Box<dyn TrustedLen<Item = T> + 'a> {
    type Output = Box<dyn TrustedLen<Item = U> + 'a>;

    #[inline]
    /// # Safety
    ///
    /// the caller must ensure T and U is actually the same type
    unsafe fn into_dtype(self) -> Self::Output {
        std::mem::transmute(self)
    }
}

pub type TvIter<'a, T> = Box<dyn TrustedLen<Item = T> + 'a>;

#[derive(GetDtype)]
pub enum DynTrustIter<'a> {
    Bool(TvIter<'a, bool>),
    F32(TvIter<'a, f32>),
    F64(TvIter<'a, f64>),
    I32(TvIter<'a, i32>),
    I64(TvIter<'a, i64>),
    U8(TvIter<'a, u8>),
    U64(TvIter<'a, u64>),
    Usize(TvIter<'a, usize>),
    String(TvIter<'a, String>),
    OptUsize(TvIter<'a, Option<usize>>),
    VecUsize(TvIter<'a, Vec<usize>>),
    #[cfg(feature = "py")]
    Object(TvIter<'a, Object>),
    #[cfg(feature = "time")]
    DateTime(TvIter<'a, DateTime>),
    #[cfg(feature = "time")]
    TimeDelta(TvIter<'a, TimeDelta>),
}

unsafe impl Send for DynTrustIter<'_> {}
unsafe impl Sync for DynTrustIter<'_> {}

impl<'a> DynTrustIter<'a> {
    #[inline]
    #[allow(unreachable_patterns)]
    pub fn collect_vec(self) -> TResult<DynVec> {
        crate::match_trust_iter!(self; dynamic(i) => Ok(i.collect_trusted_to_vec().into()),)
    }
}

macro_rules! impl_from {

    ($($(#[$meta:meta])? ($arm: ident, $ty: ty, $func_name: ident)),* $(,)?) => {
        impl<'a> DynTrustIter<'a> {
            $(
                $(#[$meta])?
                pub fn $func_name(self) -> TResult<TvIter<'a, $ty>> {
                    if let DynTrustIter::$arm(v) = self {
                        Ok(v)
                    } else {
                        tbail!("TrustIter is not of type {:?}", DataType::$arm)
                    }
            })*
        }

        impl<'a, T: GetDataType, I: TrustedLen<Item=T> + 'a> From<I> for DynTrustIter<'a> {
            #[allow(unreachable_patterns)]
            #[inline]
            fn from(iter: I) -> Self {
                match T::dtype() {
                    $(
                        $(#[$meta])? DataType::$arm => {
                            let iter: TvIter<'a, T> = Box::new(iter);
                            // safety: we have checked the type
                            unsafe{DynTrustIter::<'a>::$arm(iter.into_dtype().into())}
                        },
                    )*
                    type_ => unimplemented!("Create TrustIter from type {:?} is not implemented", type_),
                }
            }
        }
    };
}

impl_from!(
    (Bool, bool, bool),
    (F32, f32, f32),
    (F64, f64, f64),
    (I32, i32, i32),
    (I64, i64, i64),
    (U8, u8, u8),
    (U64, u64, u64),
    (Usize, usize, usize),
    (String, String, string),
    (OptUsize, Option<usize>, opt_usize),
    (VecUsize, Vec<usize>, vec_usize),
    #[cfg(feature = "py")]
    (Object, Object, object),
    #[cfg(feature = "time")]
    (DateTime, DateTime, datetime),
    #[cfg(feature = "time")]
    (TimeDelta, TimeDelta, timedelta)
);

#[macro_export]
macro_rules! match_trust_iter {
    ($($tt: tt)*) => {
        $crate::match_enum!(DynTrustIter, $($tt)*)
    };
}

#[macro_export]
/// create dynamic trust iter
macro_rules! dt_iter {
    ($($tt: tt)*) => {
        {
            let vec: $crate::prelude::DynVec = vec![ $($tt)* ].into();
            vec.into_titer().unwrap()
        }
    };
}