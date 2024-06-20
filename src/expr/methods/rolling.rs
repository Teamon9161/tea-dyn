// use crate::prelude::TrustIterCast;
use crate::prelude::*;
use tevec::ndarray::{concatenate, ArrayView1, Axis, Ix1};

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

impl Rolling {
    #[allow(unreachable_patterns)]
    pub fn agg(self, func: Expr) -> Expr {
        let node = BaseNode {
            name: "rolling_apply",
            func: Arc::new(move |data: Data, backend| {
                match data {
                    Data::Vec(_) => {
                        match_vec!(
                            data.into_vec()?;
                            Dynamic(vec) => {
                                // let func = func.to_func();
                                let iter = vec.rolling_custom_iter(self.window, |v| {
                                    let ctx = Context::new(v);
                                    let res = func.eval(&ctx, Some(Backend::Vec)).unwrap();
                                    res.into_owned(Some(backend)).unwrap()
                                }).collect_trusted_to_vec();
                                let res = concat(iter, None)?;
                                Ok(res)
                            },
                        )
                    }
                    // Data::Array(_) => {
                    //     match_array!(
                    //         data.into_array()?;
                    //         Dynamic(arr) => {
                    //             let arr1 = arr.view().into_dimensionality::<Ix1>().unwrap();
                    //             let iter = arr1.rolling_custom_iter::<'_, _, _>(
                    //                 self.window,
                    //                 |_view| {
                    //                     // let dyn_arr: DynArray = view.view().into_dyn().into();
                    //                     // let ctx = Context::new(dyn_arr);
                    //                     // let res = func(&ctx, Some(Backend::Numpy)).unwrap();
                    //                     // unsafe{std::mem::transmute::<_, Data<'static>>(res)}
                    //                     1
                    //                 },
                    //                 // None
                    //             );
                    //             // todo!();
                    //             let arr: DynArray = arr1.to_owned().into_dyn().into();
                    //             Ok(arr.into())
                    //             // let arr: DynArray = arr.into_dyn().into();
                    //             // Ok(arr.into())
                    //         },
                    //     )
                    // }
                    _ => todo!(),
                }
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
