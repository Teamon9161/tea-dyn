use std::borrow::Cow;

use tevec::ndarray::{Array1, ArrayD, ArrayView1, ArrayViewD, ArrayViewMut1, ArrayViewMutD};

use crate::prelude::*;

impl<'a> From<DynTrustIter<'a>> for Data<'a> {
    #[inline]
    fn from(iter: DynTrustIter<'a>) -> Self {
        Data::TrustIter(Arc::new(iter))
    }
}

impl<'a> From<DynArray<'a>> for Data<'a> {
    #[inline]
    fn from(arr: DynArray<'a>) -> Self {
        Data::Array(Arc::new(arr))
    }
}

impl<'a, T> From<ArrayViewD<'a, T>> for Data<'a>
where
    DynArray<'a>: From<ArbArray<'a, T>>,
{
    #[inline]
    fn from(arr: ArrayViewD<'a, T>) -> Self {
        DynArray::from(arr.into()).into()
    }
}

impl<'a, T> From<ArrayViewMutD<'a, T>> for Data<'a>
where
    DynArray<'a>: From<ArbArray<'a, T>>,
{
    #[inline]
    fn from(arr: ArrayViewMutD<'a, T>) -> Self {
        DynArray::from(arr.into()).into()
    }
}

impl<'a, T: 'a> From<ArrayD<T>> for Data<'a>
where
    DynArray<'a>: From<ArbArray<'a, T>>,
{
    #[inline]
    fn from(arr: ArrayD<T>) -> Self {
        DynArray::from(arr.into()).into()
    }
}

impl<'a, T> From<ArrayView1<'a, T>> for Data<'a>
where
    DynArray<'a>: From<ArbArray<'a, T>>,
{
    #[inline]
    fn from(arr: ArrayView1<'a, T>) -> Self {
        DynArray::from(arr.into_dyn().into()).into()
    }
}

impl<'a, T> From<ArrayViewMut1<'a, T>> for Data<'a>
where
    DynArray<'a>: From<ArbArray<'a, T>>,
{
    #[inline]
    fn from(arr: ArrayViewMut1<'a, T>) -> Self {
        DynArray::from(arr.into_dyn().into()).into()
    }
}

impl<'a, T: 'a> From<Array1<T>> for Data<'a>
where
    DynArray<'a>: From<ArbArray<'a, T>>,
{
    #[inline]
    fn from(arr: Array1<T>) -> Self {
        DynArray::from(arr.into_dyn().into()).into()
    }
}

impl<'a, T: Clone + 'a> From<Vec<T>> for Data<'a>
where
    DynVec<'a>: From<Cow<'a, [T]>>,
{
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        let vec: DynVec = vec.into();
        vec.into()
    }
}

impl<'a, T: Clone + 'a> From<&'a [T]> for Data<'a>
where
    DynVec<'a>: From<Cow<'a, [T]>>,
{
    #[inline]
    fn from(vec: &'a [T]) -> Self {
        let vec: DynVec = vec.into();
        vec.into()
    }
}

impl<'a, T: Clone> From<Cow<'a, [T]>> for Data<'a>
where
    DynVec<'a>: From<Cow<'a, [T]>>,
{
    #[inline]
    fn from(vec: Cow<'a, [T]>) -> Self {
        let vec: DynVec<'a> = vec.into();
        vec.into()
    }
}

impl<'a> From<DynVec<'a>> for Data<'a> {
    #[inline]
    fn from(vec: DynVec<'a>) -> Self {
        Data::Vec(Arc::new(vec))
    }
}
