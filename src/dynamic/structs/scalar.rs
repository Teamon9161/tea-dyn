#![allow(unreachable_patterns)]
use crate::prelude::*;
use derive_more::From;
#[cfg(feature = "pl")]
use polars::prelude::AnyValue;
use tea_macros::GetDtype;

#[derive(GetDtype, From, Debug, Clone)]
pub enum Scalar {
    Bool(bool),
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
    U8(u8),
    U64(u64),
    Usize(usize),
    String(String),
    OptBool(Option<bool>),
    OptI32(Option<i32>),
    OptI64(Option<i64>),
    OptF32(Option<f32>),
    OptF64(Option<f64>),
    OptUsize(Option<usize>),
    VecUsize(Vec<usize>),
    #[cfg(feature = "py")]
    Object(Object),
    #[cfg(feature = "time")]
    DateTimeMs(DateTime<unit::Millisecond>),
    #[cfg(feature = "time")]
    DateTimeUs(DateTime<unit::Microsecond>),
    #[cfg(feature = "time")]
    DateTimeNs(DateTime<unit::Nanosecond>),
    #[cfg(feature = "time")]
    TimeDelta(TimeDelta),
}

impl<'a> TryFrom<AnyValue<'a>> for Scalar {
    type Error = TError;
    fn try_from(value: AnyValue<'a>) -> TResult<Self> {
        match_enum!(AnyValue, value;
            (Boolean | UInt8 | UInt64 | Int32 | Int64 | Float32 | Float64)(v) => Ok(v.into()),
            String(v) => Ok(v.to_string().into()),
            // TODO: implement object, datetime, use smart string in Scalar?
            // StringOwned(s) => Ok(s.into()),
        )
    }
}

macro_rules! impl_from {

    ($($(#[$meta:meta])? ($arm: ident, $dtype: ident $(($inner: path))?, $ty: ty, $func_name: ident)),* $(,)?) => {
        impl Scalar {
            $(
                $(#[$meta])?
                pub fn $func_name(self) -> TResult<$ty> {
                    if let Scalar::$arm(v) = self {
                        Ok(v)
                    } else {
                        tbail!("Scalar is not of type {:?}", <$ty>::type_())
                    }
            })*
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
macro_rules! match_scalar {
    ($($tt: tt)*) => {
        $crate::match_enum!(Scalar, $($tt)*)
    };
}

impl Scalar {
    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn titer(&self) -> TResult<DynTrustIter> {
        if let Scalar::VecUsize(v) = self {
            // clone vector is expensive, so we use reference instead
            Ok(v.titer().into())
        } else {
            self.clone().into_titer()
        }
    }

    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn into_titer(self) -> TResult<DynTrustIter<'static>> {
        match_scalar!(self; Dynamic(v) => Ok(std::iter::once(v).into()),)
    }

    #[inline]
    pub fn cheap_clone(&self) -> Option<Self> {
        match_scalar!(
            self;
            (Normal | TimeRelated)(v) => Ok((*v).into()),
            (String | #[cfg(feature = "py")] Object)(v) => Ok(v.clone().into()),
        )
        .ok()
    }

    #[inline]
    pub fn clone_inner(&self) -> Self {
        match_scalar!(
            self;
            (Normal | TimeRelated)(v) => Ok((*v).into()),
            (String | #[cfg(feature = "py")] Object | VecUsize)(v) => Ok(v.clone().into()),
        )
        .unwrap()
    }

    #[inline]
    pub fn cast_i32(self) -> TResult<i32> {
        match_scalar!(self; Numeric(v) => Ok(v.cast()),)
    }

    #[inline]
    pub fn cast_i64(self) -> TResult<i64> {
        match_scalar!(self; Numeric(v) => Ok(v.cast()),)
    }

    #[inline]
    pub fn cast_f32(self) -> TResult<f32> {
        match_scalar!(self; Numeric(v) => Ok(v.cast()),)
    }

    #[inline]
    pub fn cast_f64(self) -> TResult<f64> {
        match_scalar!(self; Numeric(v) => Ok(v.cast()),)
    }

    #[inline]
    pub fn cast_bool(self) -> TResult<bool> {
        match_scalar!(self; Numeric(v) => Ok(v.cast()),)
    }

    #[inline]
    pub fn cast_usize(self) -> TResult<usize> {
        match_scalar!(self; Numeric(v) => Ok(v.cast()),)
    }

    #[inline]
    pub fn cast_optusize(self) -> TResult<Option<usize>> {
        match_scalar!(self; Numeric(v) => Ok(v.cast()),)
    }
}

macro_rules! impl_cast {
    ($($(#[$meta: meta])? $real: ty),*) => {
        $(
            $(#[$meta])?
            impl Cast<$real> for Scalar {
                #[inline]
                fn cast(self) -> $real {
                    match_scalar!(self; Cast(v) => Ok(v.cast()),).unwrap()
                }
            }
        )*
    };
}

impl_cast!(
    bool,
    f32,
    f64,
    i32,
    i64,
    u8,
    u64,
    usize,
    String,
    Option<bool>,
    Option<f32>,
    Option<f64>,
    Option<i32>,
    Option<i64>,
    Option<usize>,
    #[cfg(feature = "py")]
    Object,
    // #[cfg(feature = "time")]
    // DateTime,
    #[cfg(feature = "time")]
    TimeDelta
);

#[cfg(feature = "time")]
impl<U: TimeUnitTrait> Cast<DateTime<U>> for Scalar {
    #[inline]
    #[allow(clippy::useless_conversion)]
    /// we have special implemention for cast to same unit, so just ignore clippy::useless_conversion here
    fn cast(self) -> DateTime<U> {
        match_scalar!(
            self;
            DateTimeMs(v) | DateTimeUs(v) | DateTimeNs(v) => Ok(v.into_unit::<U>().into()),
        )
        .unwrap()
    }
}

impl Cast<Vec<usize>> for Scalar {
    #[inline]
    #[allow(unreachable_patterns)]
    fn cast(self) -> Vec<usize> {
        match_scalar!(
            self;
            Cast(v) => {Ok(vec![v.cast()])},
            VecUsize(v) => Ok(v.cast()),
        )
        .unwrap()
    }
}

#[macro_export]
/// create scalar
macro_rules! scalar {
    ($e: expr) => {{
        let s: $crate::prelude::Scalar = $e.into();
        s
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_get_dtype() {
        let s = Scalar::F64(1.0);
        assert_eq!(s.dtype(), DataType::F64);
    }
}
