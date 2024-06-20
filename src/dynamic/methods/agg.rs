#![allow(unreachable_patterns)]
use crate::prelude::*;

impl<'a> DynTrustIter<'a> {
    #[inline]
    pub fn vsum(self) -> TResult<Scalar> {
        match_trust_iter!(self; Numeric(e) => Ok(e.vsum().unwrap_or_default().into()),)
    }
}
