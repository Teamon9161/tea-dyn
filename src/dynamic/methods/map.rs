#![allow(unreachable_patterns)]
use crate::prelude::*;

impl<'a> DynTrustIter<'a> {
    #[inline]
    pub fn vabs(self) -> TResult<Self> {
        match_trust_iter!(self; Numeric(e) => Ok(e.vabs().into()),)
    }

    #[inline]
    pub fn abs(self) -> TResult<Self> {
        match_trust_iter!(self; PureNumeric(e) => Ok(e.abs().into()),)
    }

    #[inline]
    pub fn vshift(self, n: i32, value: Option<Scalar>) -> TResult<Self> {
        match_trust_iter!(self; Dynamic(e) => {
            Ok(e.vshift(n, value.map(|v| v.cast())).into())
        },)
    }
}
