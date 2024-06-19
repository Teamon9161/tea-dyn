use super::ArbArray;
use crate::prelude::Dtype;
use tevec::prelude::*;

impl<'a, T> ArbArray<'a, T> {
    /// Cast to another type
    #[inline]
    pub fn cast_to<T2>(self) -> ArbArray<'a, T2>
    where
        T: Dtype + Cast<T2> + Clone,
        T2: Dtype + Clone + 'a,
    {
        if T::type_() == T2::type_() {
            // safety: T and T2 are the same type
            unsafe { std::mem::transmute::<ArbArray<'a, T>, ArbArray<'a, T2>>(self) }
        } else {
            self.view().mapv(|v| v.cast()).into()
        }
    }

    #[inline]
    fn cast_with<T2>(self, _other: &ArbArray<'a, T2>) -> ArbArray<'a, T2>
    where
        T2: Dtype,
        T: Dtype,
    {
        if T::type_() == T2::type_() {
            // safety: T and T2 are the same type
            unsafe { std::mem::transmute::<ArbArray<'a, T>, ArbArray<'a, T2>>(self) }
        } else {
            unreachable!()
        }
    }

    #[inline]
    fn cast_ref_with<T2>(&self, _other: &ArbArray<'a, T2>) -> &ArbArray<'a, T2>
    where
        T2: Dtype,
        T: Dtype,
    {
        if T::type_() == T2::type_() {
            // safety: T and T2 are the same type
            unsafe { std::mem::transmute::<&ArbArray<'a, T>, &ArbArray<'a, T2>>(self) }
        } else {
            unreachable!()
        }
    }
}
