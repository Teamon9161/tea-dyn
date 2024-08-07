use crate::prelude::*;
use tevec::macros::GetDtype;
#[cfg(feature = "pl")]
use tevec::polars::prelude::{ChunkedArray, IntoSeries, Series};

pub type TvIter<'a, T> = Box<dyn TrustedLen<Item = T> + 'a>;

trait IterIntoDyn<'a> {
    type Dyn;
    fn into_dyn(self) -> Self::Dyn;
}

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
    OptBool(TvIter<'a, Option<bool>>),
    OptF32(TvIter<'a, Option<f32>>),
    OptF64(TvIter<'a, Option<f64>>),
    OptI32(TvIter<'a, Option<i32>>),
    OptI64(TvIter<'a, Option<i64>>),
    OptUsize(TvIter<'a, Option<usize>>),
    VecUsize(TvIter<'a, Vec<usize>>),
    #[cfg(feature = "py")]
    Object(TvIter<'a, Object>),
    #[cfg(feature = "time")]
    DateTimeMs(TvIter<'a, DateTime<unit::Millisecond>>),
    #[cfg(feature = "time")]
    DateTimeUs(TvIter<'a, DateTime<unit::Microsecond>>),
    #[cfg(feature = "time")]
    DateTimeNs(TvIter<'a, DateTime<unit::Nanosecond>>),
    #[cfg(feature = "time")]
    TimeDelta(TvIter<'a, TimeDelta>),
}

impl<'a> Default for DynTrustIter<'a> {
    #[inline]
    fn default() -> Self {
        DynTrustIter::Bool(Box::new(std::iter::empty().to_trust(0)))
    }
}

impl<'a> std::fmt::Debug for DynTrustIter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dtype = self.dtype();
        write!(f, "DynTrustIter({:?})", dtype)
    }
}

unsafe impl Send for DynTrustIter<'_> {}
unsafe impl Sync for DynTrustIter<'_> {}

impl<'a, I: TrustedLen + 'a> From<I> for DynTrustIter<'a>
where
    TvIter<'a, I::Item>: IterIntoDyn<'a, Dyn = DynTrustIter<'a>>,
{
    #[allow(unreachable_patterns)]
    #[inline]
    fn from(iter: I) -> Self {
        let iter: TvIter<'a, I::Item> = Box::new(iter);
        iter.into_dyn()
    }
}

// impl<'a, I: TrustedLen + 'a> IterIntoDyn<'a> for I
// where
//     DynTrustIter<'a>: From<TvIter<'a, I::Item>>,
// {
//     type Dyn = DynTrustIter<'a>;

//     #[inline]
//     fn into_dyn(self) -> Self::Dyn {
//         let iter: TvIter<'a, I::Item> = Box::new(self);
//         iter.into()
//     }
// }

impl<'a> DynTrustIter<'a> {
    #[inline]
    #[allow(unreachable_patterns)]
    pub fn collect_vec<'b>(self) -> TResult<DynVec<'b>> {
        crate::match_trust_iter!(self; Dynamic(i) => Ok(i.collect_trusted_to_vec().into()),)
    }

    #[inline]
    #[allow(unreachable_patterns)]
    pub fn collect_array<'b>(self) -> TResult<DynArray<'b>> {
        self.collect_vec()?.into_array()
    }

    #[inline]
    pub fn chain(self, other: Self) -> Self {
        match_trust_iter!(self; Cast(i) => {
            match_trust_iter!(other; Cast(j) => {
                let iter = i.chain(j.map(Cast::cast));
                Ok(iter.into())
            },)
        },)
        .unwrap()
    }

    #[inline]
    #[cfg(feature = "pl")]
    pub fn collect_series(self) -> TResult<Series> {
        use tevec::polars::datatypes::*;
        crate::match_trust_iter!(
            self;
            Bool(i) => {
                let arr: ChunkedArray<BooleanType> = i.map(|v| v.to_opt()).collect_trusted_vec1();
                Ok(arr.into_series())
            },
            F32(i) => {
                let arr: ChunkedArray<Float32Type> = i.map(|v| v.to_opt()).collect_trusted_vec1();
                Ok(arr.into_series())
            },
            F64(i) => {
                let arr: ChunkedArray<Float64Type> = i.map(|v| v.to_opt()).collect_trusted_vec1();
                Ok(arr.into_series())
            },
            I32(i) => {
                let arr: ChunkedArray<Int32Type> = i.map(|v| v.to_opt()).collect_trusted_vec1();
                Ok(arr.into_series())
            },
            I64(i) => {
                let arr: ChunkedArray<Int64Type> = i.map(|v| v.to_opt()).collect_trusted_vec1();
                Ok(arr.into_series())
            },
            OptBool(i) => {
                let arr: ChunkedArray<BooleanType> = i.collect_trusted_vec1();
                Ok(arr.into_series())
            },
            OptF32(i) => {
                let arr: ChunkedArray<Float32Type> = i.collect_trusted_vec1();
                Ok(arr.into_series())
            },
            OptF64(i) => {
                let arr: ChunkedArray<Float64Type> = i.collect_trusted_vec1();
                Ok(arr.into_series())
            },
            OptI32(i) => {
                let arr: ChunkedArray<Int32Type> = i.collect_trusted_vec1();
                Ok(arr.into_series())
            },
            OptI64(i) => {
                let arr: ChunkedArray<Int64Type> = i.collect_trusted_vec1();
                Ok(arr.into_series())
            },
        )
    }

    #[inline]
    pub fn collect<'b>(self, backend: Backend) -> TResult<Data<'b>> {
        match backend {
            Backend::Vec => self.collect_vec().map(|o| o.into()),
            Backend::Numpy | Backend::Pandas => self.collect_array().map(Into::into),
            #[cfg(feature = "pl")]
            Backend::Polars => self.collect_series().map(Into::into),
        }
    }
}

macro_rules! impl_from {

    ($($(#[$meta:meta])? ($arm: ident, $dtype: ident $(($inner: path))?, $ty: ty, $func_name: ident)),* $(,)?) => {
        impl<'a> DynTrustIter<'a> {
            $(
                $(#[$meta])?
                pub fn $func_name(self) -> TResult<TvIter<'a, $ty>> {
                    if let DynTrustIter::$arm(v) = self {
                        Ok(v)
                    } else {
                        tbail!("TrustIter is not of type {:?}", <$ty>::type_())
                    }
                }
            )*
        }


        $(
            $(#[$meta])?
            impl<'a> IterIntoDyn<'a> for TvIter<'a, $ty>{
                type Dyn = DynTrustIter<'a>;

                #[inline]
                fn into_dyn(self) -> Self::Dyn {
                    DynTrustIter::$arm(self)
                }
            }
        )*
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

pub trait TrustIterCast<'a>: TrustedLen + 'a {
    fn cast_to<T2: 'a>(self) -> Box<dyn TrustedLen<Item = T2> + 'a>
    where
        Self::Item: Cast<T2> + 'a;
    fn cast_with<I2: TrustedLen>(self, i2: &I2) -> Box<dyn TrustedLen<Item = I2::Item> + 'a>
    where
        I2::Item: 'a,
        Self::Item: Cast<I2::Item> + 'a;
}

impl<'a, I: TrustedLen + 'a> TrustIterCast<'a> for I {
    #[inline]
    // fn cast_to<T2: 'a>(self) -> impl TrustedLen<Item = T2> + 'a
    fn cast_to<T2: 'a>(self) -> Box<dyn TrustedLen<Item = T2> + 'a>
    where
        Self::Item: Cast<T2> + 'a,
    {
        Box::new(self.map(Cast::cast))
    }

    fn cast_with<I2: TrustedLen>(self, _i2: &I2) -> Box<dyn TrustedLen<Item = I2::Item> + 'a>
    where
        I2::Item: 'a,
        Self::Item: Cast<I2::Item> + 'a,
    {
        Box::new(self.map(Cast::cast))
    }
}
