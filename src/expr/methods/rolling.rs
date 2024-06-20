use crate::prelude::*;
use tevec::ndarray::{concatenate, ArrayView1, Axis, Ix1};
#[cfg(feature = "pl")]
use tevec::polars::prelude::Series;

#[derive(Clone)]
pub struct Rolling {
    window: usize,
    expr: Expr,
}

pub fn concat_iter<'a, I: IntoIterator<Item = DynTrustIter<'a>>>(is: I) -> DynTrustIter<'a> {
    let mut iter = is.into_iter();
    if let Some(mut i1) = iter.next() {
        for i in iter {
            i1 = i1.chain(i);
        }
        i1
    } else {
        Default::default()
    }

    // use tevec::polars::prelude::*;
    // Float32Chunked::from_vec(name, v)
}

pub fn concat(es: Vec<Data>, axis: Option<usize>) -> TResult<Data> {
    if es.is_empty() {
        return Ok(Default::default());
    } else if es.len() == 1 {
        return Ok(es.into_iter().next().unwrap());
    }
    // fast path for concat iter
    if Vec::get(&es, 0).unwrap().is_trust_iter() && Vec::get(&es, 1).unwrap().is_trust_iter() {
        let is = es
            .into_iter()
            .map(|e| e.try_into_iter().unwrap())
            .collect_trusted_to_vec();
        return Ok(concat_iter(is).into());
    }

    #[cfg(feature = "pl")]
    if Vec::get(&es, 0).unwrap().is_series() {
        // this is needed because we can't consume polars series and turn it into a trust iter
        let s1 = Vec::get(&es, 0).unwrap();
        let mut i1 = s1.try_titer().unwrap();
        for i in es.iter().skip(1) {
            i1 = i1.chain(i.try_titer().unwrap());
        }
        return Ok(i1.collect_series().unwrap().into());
    }
    let len = es.len();

    let mut es = es.into_iter();
    let d1 = es.next().unwrap(); // first data
    match d1.try_into_iter() {
        Ok(mut i1) => {
            for i in es {
                i1 = i1.chain(i.try_into_iter().unwrap());
            }
            Ok(i1.into())
        }
        Err(d1) => {
            // multi dimensional array
            match_array!(d1.into_array()?; Cast(a1) => {
                // cast data to the same dtype
                let mut data = Vec::with_capacity(len);
                data.push(a1);
                es.for_each(|e| {
                    match_array!(e.into_array().unwrap(); Cast(a) => {
                        data.push(a.cast_into());
                        Ok(())
                    },).unwrap();
                });
                // create a view of the data
                let arr_views = data.iter().map(|a| a.view()).collect_trusted_to_vec();
                let out = concatenate(Axis(axis.unwrap_or(0)), &arr_views).map_err(|_| terr!("concat array error"))?;
                let out: DynArray = out.into();
                Ok(out.into())
            },)
        }
    }
}

#[allow(unreachable_patterns)]
fn vec_rolling<'b>(
    vec: Arc<DynVec<'b>>,
    window: usize,
    func: &Expr,
    backend: Backend,
) -> TResult<Data<'b>> {
    match_vec!(
        vec.as_ref();
        Dynamic(vec) => {
            // let func = func.to_func();
            let iter = vec.rolling_custom_iter(window, move |v| {
                let ctx = Context::new(v);
                let res = func.eval(&ctx, Some(backend)).unwrap();
                res.into_owned(Some(backend)).unwrap()
            }).collect_trusted_to_vec();
            let res = concat(iter, None)?;
            Ok(res)
        },
    )
}

#[allow(unreachable_patterns)]
fn array_rolling<'b>(
    arr: Arc<DynArray<'b>>,
    window: usize,
    func: &Expr,
    backend: Backend,
) -> TResult<Data<'b>> {
    match_array!(
        arr.as_ref();
        Dynamic(arr) => {
            let arr1 = arr.view().into_dimensionality::<Ix1>().unwrap();
            // due to current limitations in the borrow checker, rolling_custom_iter
            // implies a `'static` lifetime, but this is actually safe
            let arr1: ArrayView1<'_, _> = unsafe{arr1.into_life()};
            let iter = arr1.rolling_custom_iter(
                window,
                move |view| {
                    let dyn_arr: DynArray = view.view().into_dyn().into();
                    let ctx = Context::new(dyn_arr);
                    let res = func.eval(&ctx, Some(backend)).unwrap();
                    res.into_owned(Some(backend)).unwrap()
                },
            ).collect_trusted_to_vec();
            let res = concat(iter, None)?;
            Ok(res)
        },
    )
}

#[cfg(feature = "pl")]
fn series_rolling<'b>(
    se: Series,
    window: usize,
    func: &Expr,
    backend: Backend,
) -> TResult<Data<'b>> {
    // use tevec::polars::prelude::*;
    match_series!(
        se,
        s;
        {
            let iter = s.rolling_custom_iter(window, move |v| {
                let series: Series = if let std::borrow::Cow::Owned(s) = v {
                    s.into()
                } else{
                    unreachable!("implement error")
                };
                let ctx = Context::new(series);
                let res = func.eval(&ctx, Some(backend)).unwrap();
                res.into_owned(Some(backend)).unwrap()
            }).collect_trusted_to_vec();
            let res = concat(iter, None)?;
            Ok(res)
        }
    )
}

impl Rolling {
    // #[allow(unreachable_patterns)]
    pub fn apply(self, func: Expr) -> Expr {
        let node = BaseNode {
            name: "rolling_apply",
            func: Arc::new(move |data: Data, backend| match data {
                Data::Vec(vec) => vec_rolling(vec, self.window, &func, backend),
                Data::TrustIter(iter) => match Arc::try_unwrap(iter) {
                    Ok(iter) => {
                        vec_rolling(iter.collect_vec()?.into(), self.window, &func, backend)
                    }
                    Err(_) => {
                        tbail!("trust iter is shared, cann't collect and rolling shared iter")
                    }
                },
                Data::Array(arr) => array_rolling(arr, self.window, &func, backend),
                Data::Scalar(_) => tbail!("rolling apply not supported for scalar"),
                #[cfg(feature = "pl")]
                Data::Series(series) => series_rolling(series, self.window, &func, backend),
            }),
        };
        self.expr.chain(node)
    }
}

impl Expr {
    #[inline]
    pub fn rolling(self, window: usize) -> Rolling {
        Rolling { window, expr: self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tevec::ndarray::arr1;
    #[test]
    fn test_rolling_apply() -> TResult<()> {
        // rolling in vec
        const LENGTH: usize = 100;
        let data: Vec<_> = (0..LENGTH).collect();
        let ctx = Context::new(data);
        let expr = s(0).rolling(100).apply(s(0).sum());
        expr.eval(&ctx, Some(Backend::Vec))?;
        // let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        // let ctx = Context::new(v);
        // let res = s(0)
        //     .rolling(3)
        //     .apply(s(0).sum())
        //     .eval(&ctx, Some(Backend::Vec))?
        //     .into_vec()?
        //     .i32()?;
        // assert_eq!(res.as_ref(), &[1, 3, 6, 9, 12, 15, 18, 21, 24]);
        // // rolling in array
        // let arr = arr1(&[1, -2, 3, -4, 5, -6, 7, 8, 9]);
        // let ctx = Context::new(arr.into_dyn());
        // let res = s(0)
        //     .rolling(3)
        //     .apply(s(0).abs().sum())
        //     .eval(&ctx, Some(Backend::Vec))?
        //     .into_vec()?
        //     .i32()?;
        // assert_eq!(res.as_ref(), &[1, 3, 6, 9, 12, 15, 18, 21, 24]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "pl")]
    fn test_rolling_apply_pl() -> TResult<()> {
        use tevec::polars::prelude::*;
        // use tevec::polars::testing::assert_series_eq;
        let expr = s(0).rolling(3).apply(s(0).sum()).alias("sum");
        // rolling in series
        let v = Series::new("a", &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let ctx = Context::new(v);
        let res = expr.eval(&ctx, Some(Backend::Vec))?.into_vec()?.i32()?;
        assert_eq!(res.as_ref(), &[1, 3, 6, 9, 12, 15, 18, 21, 24]);
        // rolling in series and return series
        let v = Series::new("a", &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let ctx = Context::new(v);
        let res = expr.eval(&ctx, Some(Backend::Polars))?.into_series()?;
        assert!(res.equals(&Series::new("sum", &[1, 3, 6, 9, 12, 15, 18, 21, 24])));
        Ok(())
    }
}
