#![allow(unreachable_patterns)]
use crate::prelude::*;
use derive_more::From;
use std::borrow::Cow;
use tevec::macros::GetDtype;
#[cfg(feature = "pl")]
use tevec::polars::prelude::Series;

impl<T, U> TransmuteDtype<U> for Vec<T> {
    type Output = Vec<U>;

    #[inline]
    /// # Safety
    ///
    /// the caller must ensure T and U is actually the same type
    unsafe fn into_dtype(self) -> Self::Output {
        std::mem::transmute(self)
    }
}

impl<'a, T, U: 'a> TransmuteDtype<U> for &'a [T] {
    type Output = &'a [U];

    #[inline]
    /// # Safety
    ///
    /// the caller must ensure T and U is actually the same type
    unsafe fn into_dtype(self) -> Self::Output {
        std::mem::transmute(self)
    }
}

#[derive(GetDtype, Debug, From)]
pub enum DynVec<'a> {
    Bool(Cow<'a, [bool]>),
    F32(Cow<'a, [f32]>),
    F64(Cow<'a, [f64]>),
    I32(Cow<'a, [i32]>),
    I64(Cow<'a, [i64]>),
    U8(Cow<'a, [u8]>),
    U64(Cow<'a, [u64]>),
    Usize(Cow<'a, [usize]>),
    String(Cow<'a, [String]>),
    OptBool(Cow<'a, [Option<bool>]>),
    OptF32(Cow<'a, [Option<f32>]>),
    OptF64(Cow<'a, [Option<f64>]>),
    OptI32(Cow<'a, [Option<i32>]>),
    OptI64(Cow<'a, [Option<i64>]>),
    OptUsize(Cow<'a, [Option<usize>]>),
    VecUsize(Cow<'a, [Vec<usize>]>),
    #[cfg(feature = "py")]
    Object(Cow<'a, [Object]>),
    #[cfg(feature = "time")]
    DateTimeMs(Cow<'a, [DateTime<unit::Millisecond>]>),
    #[cfg(feature = "time")]
    DateTimeUs(Cow<'a, [DateTime<unit::Microsecond>]>),
    #[cfg(feature = "time")]
    DateTimeNs(Cow<'a, [DateTime<unit::Nanosecond>]>),
    #[cfg(feature = "time")]
    TimeDelta(Cow<'a, [TimeDelta]>),
}

impl<'a> Default for DynVec<'a> {
    #[inline]
    fn default() -> Self {
        DynVec::F64(Cow::Owned(Vec::default()))
    }
}

macro_rules! impl_from {

    ($($(#[$meta:meta])? ($arm: ident, $dtype: ident $(($inner: path))?, $ty: ty, $func_name: ident)),* $(,)?) => {
        impl<'a> DynVec<'a> {
            $(
                $(#[$meta])?
                pub fn $func_name(self) -> TResult<Cow<'a, [$ty]>> {
                    if let DynVec::$arm(v) = self {
                        Ok(v)
                    } else {
                        tbail!("Vector is not of type {:?}", <$ty>::type_())
                    }
            })*
        }

        impl<'a, T: Dtype> From<Vec<T>> for DynVec<'a> {
            #[allow(unreachable_patterns)]
            #[inline]
            fn from(vec: Vec<T>) -> Self {
                match T::type_() {
                    $(
                        $(#[$meta])? DataType::$dtype $(($inner))? => {
                            // safety: we have checked the type
                            unsafe{DynVec::$arm(vec.into_dtype().into())}
                        },
                    )*
                    type_ => unimplemented!("Create Vector from type {:?} is not implemented", type_),
                }
            }
        }

        impl<'a, T: Dtype> From<&'a [T]> for DynVec<'a> {
            #[allow(unreachable_patterns)]
            #[inline]
            fn from(vec: &'a [T]) -> Self {
                match T::type_() {
                    $(
                        $(#[$meta])? DataType::$dtype $(($inner))? => {
                            // safety: we have checked the type
                            unsafe{DynVec::$arm(Cow::Borrowed(vec.into_dtype()))}
                        },
                    )*
                    type_ => unimplemented!("Create Vector from type {:?} is not implemented", type_),
                }
            }
        }
    };
}

impl_from!(
    (Bool, Bool, bool, bool),
    (F32, F32, f32, f32),
    (F64, F64, f64, f64),
    (I32, I32, i32, i32),
    (I64, I64, i64, i64),
    (U8, U8, u8, u8),
    (U64, U64, u64, u64),
    (Usize, Usize, usize, usize),
    (String, String, String, string),
    (OptBool, OptBool, Option<bool>, opt_bool),
    (OptF32, OptF32, Option<f32>, opt_f32),
    (OptF64, OptF64, Option<f64>, opt_f64),
    (OptI32, OptI32, Option<i32>, opt_i32),
    (OptI64, OptI64, Option<i64>, opt_i64),
    (OptUsize, OptUsize, Option<usize>, opt_usize),
    (VecUsize, VecUsize, Vec<usize>, vec_usize),
    #[cfg(feature = "py")]
    (Object, Object, Object, object),
    #[cfg(feature = "time")]
    (DateTimeMs, DateTime(TimeUnit::Millisecond), DateTime<unit::Millisecond>, datetime_ms),
    #[cfg(feature = "time")]
    (DateTimeUs, DateTime(TimeUnit::Microsecond), DateTime<unit::Microsecond>, datetime_us),
    #[cfg(feature = "time")]
    (DateTimeNs, DateTime(TimeUnit::Nanosecond), DateTime<unit::Nanosecond>, datetime_ns),
    #[cfg(feature = "time")]
    (TimeDelta, TimeDelta, TimeDelta, timedelta)
);

#[macro_export]
macro_rules! match_vec {
    ($($tt: tt)*) => {
        $crate::match_enum!(DynVec, $($tt)*)
    };
}

#[macro_export]
/// create dynamic iter
macro_rules! d_vec {
    ($($tt: tt)*) => {
        {
            let vec: $crate::prelude::DynVec = vec![ $($tt)* ].into();
            vec
        }
    };
}

impl<'a> DynVec<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        match_vec!(self; Dynamic(v) => Ok(v.len()),).unwrap()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(&self, index: usize) -> TResult<Scalar> {
        match_vec!(self; Dynamic(v) => Vec1View::get(v.as_ref(), index).map(|v| v.into()),)
    }

    #[inline]
    pub fn into_owned<'b>(self) -> DynVec<'b> {
        match_vec!(self; Dynamic(v) => Ok(v.into_owned().into()),).unwrap()
    }

    #[inline]
    pub fn clone_inner<'b>(&self) -> DynVec<'b> {
        match_vec!(self; Dynamic(v) => match v {
            Cow::Borrowed(v) => Ok(v.to_vec().into()),
            Cow::Owned(v) => Ok(v.clone().into()),
        },)
        .unwrap()
    }

    #[inline]
    pub fn titer(&self) -> TResult<DynTrustIter> {
        match_vec!(self; Dynamic(v) => Ok(v.titer().into()),)
    }

    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn into_titer(self) -> TResult<DynTrustIter<'a>> {
        // match_vec!(self; Dynamic(v) => Ok(v.into_iter().into()),)
        match_vec!(self; Dynamic(v) => {
            match v {
                Cow::Owned(v) => Ok(v.into_iter().into()),
                Cow::Borrowed(v) => Ok(v.titer().into()),
            }
        },)
    }

    #[inline]
    #[cfg(feature = "ndarray")]
    pub fn into_array(self) -> TResult<DynArray<'a>> {
        use tevec::ndarray::Array1;
        match_vec!(self; Dynamic(v) => {
            match v {
                Cow::Owned(v) => {
                    let arr = Array1::from_vec(v).into_dyn();
                    Ok(arr.into())
                },
                Cow::Borrowed(v) => {
                    // TODO: can we avoid clone here?
                    let arr = Array1::from_vec(v.to_vec()).into_dyn();
                    Ok(arr.into())
                },
            }
        },)
    }

    #[inline]
    #[cfg(feature = "pl")]
    pub fn into_series(self) -> TResult<Series> {
        use tevec::{
            polars::prelude::*,
            polars_arrow::{bitmap::Bitmap, legacy::utils::CustomIterTools},
        };
        match_vec!(self;
            // zero copy
            (I32 | I64 | U64)(v) => {
                match v {
                    Cow::Owned(v) => Ok(Series::from_vec("", v)),
                    Cow::Borrowed(v) => Ok(Series::from_vec("", v.to_vec())),
                }
            },
            // zero copy but need mask
            (F32)(v) => {
                let mask_iter = v.titer().map(|v| v.is_none());
                let bitmap: Bitmap = mask_iter.collect_trusted();
                match v {
                    Cow::Owned(v) => {
                        Ok(Float32Chunked::from_vec_validity("", v, Some(bitmap)).into())
                    },
                    Cow::Borrowed(v) => Ok(Float32Chunked::from_vec_validity("", v.to_vec(), Some(bitmap)).into()),
                }
            },
            // zero copy but need mask
            (F64)(v) => {
                let mask_iter = v.titer().map(|v| v.is_none());
                let bitmap: Bitmap = mask_iter.collect_trusted();
                match v {
                    Cow::Owned(v) => {
                        Ok(Float64Chunked::from_vec_validity("", v, Some(bitmap)).into())
                    },
                    Cow::Borrowed(v) => Ok(Float64Chunked::from_vec_validity("", v.to_vec(), Some(bitmap)).into()),
                }
            },
            // clone is needed
            (Bool | PlOpt)(v) => Ok(Series::from_iter(v.titer())),
        )
    }

    #[inline]
    pub fn into_backend(self, backend: Backend) -> TResult<Data<'a>> {
        match backend {
            Backend::Vec => Ok(self.into()),
            Backend::Numpy | Backend::Pandas => self.into_array().map(Into::into),
            #[cfg(feature = "pl")]
            Backend::Polars => self.into_series().map(Into::into),
        }
    }

    #[inline]
    #[cfg(feature = "pl")]
    pub fn to_series(&self) -> TResult<Series> {
        use tevec::polars::prelude::*;
        match_vec!(self;
            (PlOpt | PlInt | Float | Bool)(v) => Ok(Series::from_iter(v.titer())),
        )
    }
}
