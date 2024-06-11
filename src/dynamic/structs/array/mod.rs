#![allow(unreachable_patterns)]
mod ndarray_ext;

pub use ndarray_ext::NdArrayExt;

use crate::prelude::*;
use derive_more::From;
#[cfg(all(feature = "py", feature = "time"))]
use numpy::datetime::{units, Datetime as NPDatetime};
use tea_macros::GetDtype;
use tevec::ndarray::prelude::*;

#[cfg(all(feature = "py", feature = "time"))]
impl Dtype for NPDatetime<units::Milliseconds> {
    #[inline]
    fn type_() -> DataType {
        DataType::DateTime(TimeUnit::Millisecond)
    }
}

#[cfg(all(feature = "py", feature = "time"))]
impl Dtype for NPDatetime<units::Microseconds> {
    #[inline]
    fn type_() -> DataType {
        DataType::DateTime(TimeUnit::Microsecond)
    }
}

#[cfg(all(feature = "py", feature = "time"))]
impl Dtype for NPDatetime<units::Nanoseconds> {
    #[inline]
    fn type_() -> DataType {
        DataType::DateTime(TimeUnit::Nanosecond)
    }
}

#[derive(From)]
pub enum ArbArray<'a, T> {
    Owned(ArrayD<T>),
    View(ArrayViewD<'a, T>),
    ViewMut(ArrayViewMutD<'a, T>),
}

#[derive(From, GetDtype)]
pub enum DynArray<'a> {
    Bool(ArbArray<'a, bool>),
    F32(ArbArray<'a, f32>),
    F64(ArbArray<'a, f64>),
    I32(ArbArray<'a, i32>),
    I64(ArbArray<'a, i64>),
    U8(ArbArray<'a, u8>),
    U64(ArbArray<'a, u64>),
    Usize(ArbArray<'a, usize>),
    String(ArbArray<'a, String>),
    OptUsize(ArbArray<'a, Option<usize>>),
    VecUsize(ArbArray<'a, Vec<usize>>),
    #[cfg(feature = "py")]
    Object(ArbArray<'a, Object>),
    #[cfg(feature = "time")]
    DateTimeMs(ArbArray<'a, DateTime<unit::Millisecond>>),
    #[cfg(feature = "time")]
    DateTimeUs(ArbArray<'a, DateTime<unit::Microsecond>>),
    #[cfg(feature = "time")]
    DateTimeNs(ArbArray<'a, DateTime<unit::Nanosecond>>),
    #[cfg(feature = "time")]
    TimeDelta(ArbArray<'a, TimeDelta>),
}

#[cfg(all(feature = "py", feature = "time"))]
impl<'a> From<ArbArray<'a, NPDatetime<units::Milliseconds>>> for DynArray<'a> {
    #[inline]
    fn from(a: ArbArray<'a, NPDatetime<units::Milliseconds>>) -> Self {
        // safety: datetime and npdatetime has the same size
        let a: ArbArray<'a, DateTime<unit::Millisecond>> = unsafe { a.into_dtype() };
        DynArray::DateTimeMs(a)
    }
}

#[cfg(all(feature = "py", feature = "time"))]
impl<'a> From<ArbArray<'a, NPDatetime<units::Microseconds>>> for DynArray<'a> {
    #[inline]
    fn from(a: ArbArray<'a, NPDatetime<units::Microseconds>>) -> Self {
        // safety: datetime and npdatetime has the same size
        let a: ArbArray<'a, DateTime<unit::Microsecond>> = unsafe { a.into_dtype() };
        DynArray::DateTimeUs(a)
    }
}

#[cfg(all(feature = "py", feature = "time"))]
impl<'a> From<ArbArray<'a, NPDatetime<units::Nanoseconds>>> for DynArray<'a> {
    #[inline]
    fn from(a: ArbArray<'a, NPDatetime<units::Nanoseconds>>) -> Self {
        // safety: datetime and npdatetime has the same size
        let a: ArbArray<'a, DateTime<unit::Nanosecond>> = unsafe { a.into_dtype() };
        DynArray::DateTimeNs(a)
    }
}

impl<'a, T, U: 'a> TransmuteDtype<U> for ArbArray<'a, T> {
    type Output = ArbArray<'a, U>;

    #[inline]
    /// # Safety
    ///
    /// the caller must ensure T and U is actually the same type
    unsafe fn into_dtype(self) -> Self::Output {
        std::mem::transmute(self)
    }
}

macro_rules! impl_from {

    ($($(#[$meta:meta])? ($arm: ident, $dtype: ident $(($inner: path))?, $ty: ty, $func_name: ident)),* $(,)?) => {
        impl<'a> DynArray<'a> {
            $(
                $(#[$meta])?
                pub fn $func_name(self) -> TResult<ArbArray<'a, $ty>> {
                    if let DynArray::$arm(v) = self {
                        Ok(v)
                    } else {
                        tbail!("DynArray is not of type {:?}", <$ty>::type_())
                    }
                }
            )*
        }

        impl<'a, T: Dtype + 'a> From<ArrayD<T>> for DynArray<'a> {
            #[allow(unreachable_patterns)]
            #[inline]
            fn from(a: ArrayD<T>) -> Self {
                match T::type_() {
                    $(
                        $(#[$meta])? DataType::$dtype $(($inner))? => {
                            // safety: we have checked the type
                            let a: ArbArray<'a, _> = a.into();
                            unsafe{DynArray::$arm(a.into_dtype().into())}
                        },
                    )*
                    type_ => unimplemented!("Create DynArray from type {:?} is not implemented", type_),
                }
            }
        }

        impl<'a, T: Dtype + 'a> From<ArrayViewD<'a, T>> for DynArray<'a> {
            #[allow(unreachable_patterns)]
            #[inline]
            fn from(a: ArrayViewD<'a, T>) -> Self {
                match T::type_() {
                    $(
                        $(#[$meta])? DataType::$dtype $(($inner))? => {
                            // safety: we have checked the type
                            let a: ArbArray<'a, _> = a.into();
                            unsafe{DynArray::$arm(a.into_dtype().into())}
                        },
                    )*
                    type_ => unimplemented!("Create DynArray from type {:?} is not implemented", type_),
                }
            }
        }

        impl<'a, T: Dtype + 'a> From<ArrayViewMutD<'a, T>> for DynArray<'a> {
            #[allow(unreachable_patterns)]
            #[inline]
            fn from(a: ArrayViewMutD<'a, T>) -> Self {
                match T::type_() {
                    $(
                        $(#[$meta])? DataType::$dtype $(($inner))? => {
                            // safety: we have checked the type
                            let a: ArbArray<'a, _> = a.into();
                            unsafe{DynArray::$arm(a.into_dtype().into())}
                        },
                    )*
                    type_ => unimplemented!("Create DynArray from type {:?} is not implemented", type_),
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
macro_rules! match_array {
    ($($tt: tt)*) => {
        $crate::match_enum!(DynArray, $($tt)*)
    };
}

#[macro_export]
macro_rules! match_arb {
    ($($tt: tt)*) => {
        $crate::match_enum!(ArbArray, $($tt)*)
    };
}

#[macro_export]
/// create dynamic array of dim1
macro_rules! d1_array {
    ($($tt: tt)*) => {
        {
            let vec: $crate::prelude::DynArray = $crate::ndarray::arr1(& [$($tt)*]).into_dimensionality::<$crate::ndarray::IxDyn>().unwrap().into();
            vec
        }
    };
}

#[macro_export]
/// create dynamic array of dim2
macro_rules! d2_array {
    ($($tt: tt)*) => {
        {
            let vec: $crate::prelude::DynArray = $crate::ndarray::arr2(& [$($tt)*]).into_dimensionality::<$crate::ndarray::IxDyn>().unwrap().into();
            vec
        }
    };
}

impl<'a, T: Clone> ArbArray<'a, T> {
    #[inline]
    pub fn len(&self) -> usize {
        match_arb!(self; Owned(v) | View(v) | ViewMut(v) => Ok(v.len()),).unwrap()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        match_arb!(self; Owned(v) | View(v) | ViewMut(v) => Ok(v.ndim()),).unwrap()
    }

    #[inline]
    pub fn get(&self, index: usize) -> TResult<&T> {
        match_arb!(self; Owned(v) | View(v) | ViewMut(v) => v.get(index).ok_or_else(|| terr!(io(index, v.len()))),)
    }

    #[inline]
    pub fn view(&self) -> ArrayViewD<'_, T> {
        match_arb!(self; Owned(v) | View(v) | ViewMut(v) => Ok(v.view()),).unwrap()
    }

    #[inline]
    /// # Safety
    ///
    /// this is safe only when 'b is actually longer than 'a
    /// do not use this function unless you are sure about the lifetime
    pub unsafe fn into_life<'b>(self) -> ArbArray<'b, T> {
        std::mem::transmute(self)
    }

    pub fn into_vec(self) -> TResult<Vec<T>> {
        if self.ndim() > 1 {
            tbail!("Array with ndim > 1 should not be converted into vector")
        }
        match self {
            ArbArray::Owned(v) => Ok(v.into_raw_vec()),
            ArbArray::View(v) => Ok(v.to_owned().into_raw_vec()),
            // TODO: can we optimize this? mut reference can be converted into owned without cloning
            ArbArray::ViewMut(v) => Ok(v.to_owned().into_raw_vec()),
        }
    }

    #[inline]
    pub fn titer<'b>(&'b self) -> TResult<Box<dyn TrustedLen<Item = T> + 'b>> {
        if self.ndim() == 1 {
            match_arb!(self; Owned(v) | View(v) | ViewMut(v) => {
                let v = v.view().into_dimensionality::<Ix1>().unwrap();
                let iter: Box<dyn TrustedLen<Item = T>> = Box::new(v.titer());
                // this is safe as data lives longer than 'a, and 'a is longer than 'b
                // and drop v will not drop the data in memory
                let iter: Box<dyn TrustedLen<Item = T> + 'b> = unsafe { std::mem::transmute(iter) };
                Ok(iter)
            },)
        } else if self.ndim() == 0 {
            match_arb!(self; Owned(v) | View(v) | ViewMut(v) => {
                let scalar = v.view().into_dimensionality::<Ix0>().unwrap().into_scalar();
                Ok(Box::new(std::iter::once(scalar.clone())))
            },)
        } else {
            tbail!("Array with ndim > 1 cannot be converted into iterator")
        }
    }

    #[inline]
    pub fn into_titer(self) -> TResult<Box<dyn TrustedLen<Item = T> + 'a>> {
        if self.ndim() == 1 {
            match_arb!(self;
                Owned(v) => {
                    let len = v.len();
                    Ok(Box::new(v.into_iter().to_trust(len)))
                },
                View(v) => {
                    let len = v.len();
                    Ok(Box::new(v.into_iter().cloned().to_trust(len)))
                },
                ViewMut(v) => {
                    let len = v.len();
                    // TODO: maybe we can use mem::take here? will it be faster?
                    Ok(Box::new(v.into_iter().map(|v| v.clone()).to_trust(len)))
                },
            )
        } else if self.ndim() == 0 {
            match_arb!(self;
                Owned(v) => {
                    let scalar = v.into_dimensionality::<Ix0>().unwrap().into_scalar();
                    Ok(Box::new(std::iter::once(scalar.clone())))
                },
                View(v) | ViewMut(v) => {
                    // TODO: can view mut case can be optimized?
                    // may be faster in case where scalar is a vec or something large
                    let scalar = v.view().into_dimensionality::<Ix0>().unwrap().into_scalar();
                    Ok(Box::new(std::iter::once(scalar.clone())))
                },
            )
        } else {
            tbail!("Array with ndim > 1 cannot be converted into iterator")
        }
    }
}

impl<'a> DynArray<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        match_array!(self; dynamic(v) => Ok(v.len()),).unwrap()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        match_array!(self; dynamic(v) => Ok(v.ndim()),).unwrap()
    }

    #[inline]
    #[allow(clippy::clone_on_copy)]
    pub fn get(&self, index: usize) -> TResult<Scalar> {
        match_array!(self; dynamic(v) => v.get(index).map(|v| v.clone().into()),)
    }

    #[inline]
    pub fn view(&self) -> DynArray<'_> {
        match_array!(self; dynamic(v) => Ok(v.view().into()),).unwrap()
    }

    #[inline]
    pub fn into_vec(self) -> TResult<DynVec> {
        match_array!(self; dynamic(v) => Ok(v.into_vec()?.into()),)
    }

    #[inline]
    pub fn from_vec(vec: DynVec) -> TResult<DynArray<'a>> {
        match_vec!(vec; dynamic(v) => {
            Ok(Array1::from_vec(v).into_dyn().into())
        },)
    }

    #[inline]
    pub fn titer(&self) -> TResult<DynTrustIter> {
        match_array!(self; dynamic(v) => Ok(v.titer()?.into()),)
    }

    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn into_titer(self) -> TResult<DynTrustIter<'a>> {
        match_array!(self; dynamic(v) => Ok(v.into_titer()?.into()),)
    }
}