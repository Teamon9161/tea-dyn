#![allow(unreachable_patterns)]
use crate::prelude::*;
use tea_macros::GetDtype;

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

#[derive(GetDtype, Debug, Clone)]
pub enum DynVec {
    Bool(Vec<bool>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    U8(Vec<u8>),
    U64(Vec<u64>),
    Usize(Vec<usize>),
    String(Vec<String>),
    OptBool(Vec<Option<bool>>),
    OptF32(Vec<Option<f32>>),
    OptF64(Vec<Option<f64>>),
    OptI32(Vec<Option<i32>>),
    OptI64(Vec<Option<i64>>),
    OptUsize(Vec<Option<usize>>),
    VecUsize(Vec<Vec<usize>>),
    #[cfg(feature = "py")]
    Object(Vec<Object>),
    #[cfg(feature = "time")]
    DateTimeMs(Vec<DateTime<unit::Millisecond>>),
    #[cfg(feature = "time")]
    DateTimeUs(Vec<DateTime<unit::Microsecond>>),
    #[cfg(feature = "time")]
    DateTimeNs(Vec<DateTime<unit::Nanosecond>>),
    #[cfg(feature = "time")]
    TimeDelta(Vec<TimeDelta>),
}

macro_rules! impl_from {

    ($($(#[$meta:meta])? ($arm: ident, $dtype: ident $(($inner: path))?, $ty: ty, $func_name: ident)),* $(,)?) => {
        impl DynVec {
            $(
                $(#[$meta])?
                pub fn $func_name(self) -> TResult<Vec<$ty>> {
                    if let DynVec::$arm(v) = self {
                        Ok(v)
                    } else {
                        tbail!("Vector is not of type {:?}", <$ty>::type_())
                    }
            })*
        }

        impl<T: GetDataType> From<Vec<T>> for DynVec {
            #[allow(unreachable_patterns)]
            #[inline]
            fn from(vec: Vec<T>) -> Self {
                match T::dtype() {
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

impl DynVec {
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
        match_vec!(self; Dynamic(v) => v.get(index).map(|v| v.into()),)
    }

    #[inline]
    pub fn titer(&self) -> TResult<DynTrustIter> {
        match_vec!(self; Dynamic(v) => Ok(v.titer().into()),)
    }

    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn into_titer(self) -> TResult<DynTrustIter<'static>> {
        match_vec!(self; Dynamic(v) => Ok(v.into_iter().into()),)
    }

    #[inline]
    #[cfg(feature = "ndarray")]
    pub fn into_array<'a>(self) -> TResult<DynArray<'a>> {
        use tevec::ndarray::Array1;
        match_vec!(self; Dynamic(v) => {
            let arr = Array1::from_vec(v).into_dyn();
            Ok(arr.into())
        },)
    }
}
