use crate::prelude::*;
use tevec::ndarray::{prelude::*, Data, DataMut, Zip};

pub trait ArrayViewExt<T, D: Dimension> {
    /// # Safety
    ///
    /// this is safe only when 'b is actually longer than 'a
    /// do not use this function unless you are sure about the lifetime
    unsafe fn into_life<'b>(self) -> ArrayView<'b, T, D>;
}

impl<T, D: Dimension> ArrayViewExt<T, D> for ArrayView<'_, T, D> {
    /// # Safety
    ///
    /// this is safe only when 'b is actually longer than 'a
    /// do not use this function unless you are sure about the lifetime
    unsafe fn into_life<'b>(self) -> ArrayView<'b, T, D> {
        std::mem::transmute(self)
    }
}

pub trait NdArrayExt<T, D: Dimension> {
    fn apply_along_axis<'a, 'b, S2, T2, F>(
        &'a self,
        out: &'b mut ArrayBase<S2, D>,
        axis: Axis,
        par: bool,
        f: F,
    ) where
        T: Send + Sync + 'a,
        T2: Send + Sync + 'b,
        S2: DataMut<Elem = T2>,
        F: Fn(ArrayView1<'a, T>, ArrayViewMut1<'b, T2>) + Send + Sync;

    fn calc_map_trust_iter_func<'a, F, U: Send + Sync + Clone>(
        &'a self,
        f: F,
        axis: Option<usize>,
        par: Option<bool>,
    ) -> ArrayD<U>
    where
        T: 'a,
        F: Fn(ArrayView1<'a, T>) -> Box<dyn TrustedLen<Item = U> + 'a> + Send + Sync;
}

impl<T: Send + Sync, S: Data<Elem = T>, D: Dimension> NdArrayExt<T, D> for ArrayBase<S, D> {
    fn apply_along_axis<'a, 'b, S2, T2, F>(
        &'a self,
        out: &'b mut ArrayBase<S2, D>,
        axis: Axis,
        par: bool,
        f: F,
    ) where
        T: Send + Sync + 'a,
        T2: Send + Sync + 'b,
        S2: DataMut<Elem = T2>,
        F: Fn(ArrayView1<'a, T>, ArrayViewMut1<'b, T2>) + Send + Sync,
    {
        if self.is_empty() || self.len_of(axis) == 0 {
            return;
        }
        let ndim = self.ndim();
        if ndim == 1 {
            let view = self.view().into_dimensionality::<Ix1>().unwrap();
            f(view, out.view_mut().into_dimensionality::<Ix1>().unwrap());
            return;
        }
        let arr_zip = Zip::from(self.lanes(axis)).and(out.lanes_mut(axis));
        if !par || (ndim == 1) {
            // non-parallel
            arr_zip.for_each(f);
        } else {
            // parallel
            arr_zip.par_for_each(f);
        }
    }

    /// use trust_iter map function on each dimension of ndarray
    fn calc_map_trust_iter_func<'a, F, U: Send + Sync + Clone>(
        &'a self,
        f: F,
        axis: Option<usize>,
        par: Option<bool>,
    ) -> ArrayD<U>
    where
        T: 'a,
        F: Fn(ArrayView1<'a, T>) -> Box<dyn TrustedLen<Item = U> + 'a> + Send + Sync,
    {
        let axis = axis.unwrap_or(0);
        let par = par.unwrap_or(false);
        let f_flag = self.is_standard_layout();
        let shape = self.raw_dim().into_shape().set_f(f_flag);
        let mut out_arr = Array::<U, D>::uninit(shape);
        let mut out_wr = out_arr.view_mut();
        let axis = Axis(axis);
        if self.len_of(axis) == 0 {
            // we don't need to do anything
        } else {
            // we don't need a fast path for dim1, as
            // dim1 will use iterator directly
            self.apply_along_axis(&mut out_wr, axis, par, move |x_1d, mut out_1d| {
                let iter = f(x_1d);
                out_1d.write_trust_iter(iter).unwrap();
            });
        }
        unsafe { out_arr.assume_init() }
            .into_dimensionality()
            .unwrap()
    }
}
