use crate::prelude::*;
use derive_more::From;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(From, Clone)]
// #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]

pub enum Node {
    Lit(LitNode),
    Select(SelectNode),
    Base(BaseNode),
    Context(CtxNode),
}

#[derive(Clone)]
// #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LitNode {
    pub value: Arc<Scalar>,
}

impl LitNode {
    #[inline]
    // we clone the scalar each time we evaluate it
    // so we return a Data which has arbitrary lifetime
    pub fn eval<'a>(&self) -> TResult<Data<'a>> {
        let res = (*self.value).clone();
        Ok(res.into())
    }
}

#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SelectNode {
    pub symbol: Symbol,
}

impl SelectNode {
    #[inline]
    pub fn select<'b>(&self, ctx: &Context<'b>) -> TResult<Data<'b>> {
        ctx.get(self.symbol.clone()).cloned()
    }

    #[inline]
    pub fn name(&self) -> Option<&str> {
        self.symbol.name()
    }
}

impl<T: Into<Symbol>> From<T> for SelectNode {
    #[inline]
    fn from(sym: T) -> Self {
        Self { symbol: sym.into() }
    }
}

#[derive(Clone)]
#[allow(clippy::type_complexity)]
// #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BaseNode {
    pub name: &'static str,
    pub func: Arc<dyn for<'a> Fn(Data<'a>, Backend) -> TResult<Data<'a>>>,
}

#[derive(Clone)]
#[allow(clippy::type_complexity)]
// the node also require context to execute other expressions
pub struct CtxNode {
    pub name: &'static str,
    pub func: Arc<dyn for<'a> Fn(Data<'a>, &Context, Backend) -> TResult<Data<'a>>>,
}
