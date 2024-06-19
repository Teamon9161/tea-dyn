use crate::prelude::TrustIterCast;
use crate::prelude::*;
use tevec::ndarray::{concatenate, Axis};

#[derive(Clone)]
pub struct Rolling {
    window: usize,
    expr: Expr,
}

pub fn concat<'a>(es: Vec<Data>, axis: Option<usize>) -> TResult<Data<'a>> {
    if es.is_empty() {
        return Ok(Default::default());
    }
    let len = es.len();
    // let e1 = unsafe { es.get_unchecked(0) };
    let mut es = es.into_iter();
    let d1 = es.next().unwrap(); // first data
    match d1.try_into_iter() {
        Ok(d1) => {
            match_trust_iter!(d1; Cast(d1) => {
                let mut res_len = d1.len();
                let mut iter = d1;
                for e in es {
                    iter = match_trust_iter!(
                        e.try_into_iter().unwrap(); 
                        Cast(i) => {
                            len += i.len();
                            Ok(iter.chain(i.cast_to()))
                        },
                    ).unwrap();
                }
                let iter: DynTrustIter = iter.to_trust(res_len).into();
                Ok(iter.into())
            },)
        }
        Err(d1) => {
            // multi dimensional array
            match_array!(d1.into_array()?; Cast(a1) => {
                let mut data = Vec::with_capacity(len);
                data.push(a1.view());
                es.for_each(|e| {
                    match_array!(e.into_array().unwrap(); Cast(a) => Ok(data.push(a.cast_to().view())),);
                });
                let out = concatenate(Axis(axis.unwrap_or(0)), &data).map_err(|_| terr!("concat array error"))?;
                let out: DynArray = out.into();
                Ok(out.into())
            },)
        }
    }
}

impl Rolling {
    // pub fn agg(self, func: Expr) -> Expr {
    //     let node = CtxNode {
    //         name: "rolling_apply",
    //         func: Arc::new(move |data, ctx, backend| {
    //             let func = func.to_func();
    //             match data {
    //                 Data::Array(_) => {
    //                     match_array!(
    //                         data.into_array()?;
    //                         Dynamic(arr) => {
    //                             let arr1: ArrayView1<_> = arr.view().into_dimensionality().unwrap();
    //                             // let iter = arr1.rolling_custom_iter(
    //                             //         self.window,
    //                             //         move |view| {
    //                             //             let dyn_arr: DynArray = view.view().into_dyn().into();
    //                             //             let ctx = Context::new(dyn_arr);
    //                             //             let res = func(&ctx, Some(Backend::Numpy)).unwrap();
    //                             //             res
    //                             //         }
    //                             //     );
    //                             todo!();
    //                             // let arr: DynArray = arr.into_dyn().into();
    //                             // Ok(arr.into())
    //                         },
    //                     )
    //                 }
    //                 _ => todo!(),
    //             }
    //         }),
    //     };
    //     self.expr.chain(node)
    // }
}

impl Expr {
    #[inline]
    pub fn rolling(self, window: usize) -> Rolling {
        Rolling { window, expr: self }
    }
}
