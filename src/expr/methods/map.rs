#![allow(unreachable_patterns)]
use crate::prelude::*;

impl Expr {
    pub fn abs(self) -> Self {
        let node = BaseNode {
            name: "abs",
            func: Arc::new(|data, backend| match data.try_into_iter() {
                Ok(iter) => Ok(iter.abs()?.into()),
                Err(data) => {
                    // TODO: should respect backend of data, or have a way to specify backend?
                    // this is needed because polars array will always fail in try_into_iter
                    if let Ok(iter) = data.try_titer() {
                        return iter.abs()?.collect(backend);
                    }
                    match_array!(
                        data.into_array()?;
                        PureNumeric(arr) => {
                            let arr: DynArray = arr.view().map(|a| a.abs()).into();
                            Ok(arr.into())
                        },
                    )
                }
            }),
        };
        self.chain(node)
    }

    pub fn vabs(self) -> Self {
        let node = BaseNode {
            name: "vabs",
            func: Arc::new(|data, backend| match data.try_into_iter() {
                Ok(iter) => Ok(iter.vabs()?.into()),
                Err(data) => {
                    if let Ok(iter) = data.try_titer() {
                        return iter.vabs()?.collect(backend);
                    }
                    match_array!(
                        data.into_array()?;
                        Numeric(arr) => {
                            let arr: DynArray = arr.view().map(|a| a.vabs()).into();
                            Ok(arr.into())
                        },
                    )
                }
            }),
        };
        self.chain(node)
    }

    pub fn vshift(
        self,
        n: Expr,
        value: Option<Expr>,
        axis: Option<usize>,
        par: Option<bool>,
    ) -> Self {
        let node = CtxNode {
            name: "shift",
            func: Arc::new(move |data, ctx, backend| {
                let n = n.eval(ctx, None)?.into_scalar()?.i32()?;
                let value = value
                    .as_ref()
                    .map(|v| v.eval(ctx, None).unwrap().into_scalar().unwrap());
                match data.try_into_iter() {
                    Ok(iter) => Ok(iter.vshift(n, value)?.into()),
                    Err(data) => {
                        if let Ok(iter) = data.try_titer() {
                            return iter.vshift(n, value)?.collect(backend);
                        }
                        match_array!(
                            data.into_array()?;
                            Dynamic(arr) => {
                                let arr: DynArray =
                                    arr.view()
                                    .calc_map_trust_iter_func(move |a| {
                                        a.into_titer().cloned().vshift(n, value.clone().map(|v| v.cast()))
                                    }
                                    , axis, par)
                                    .into();
                                Ok(arr.into())
                            },
                        )
                    }
                }
            }),
        };
        self.chain(node)
    }
}
