#![allow(unreachable_patterns)]
use crate::prelude::*;

impl<'a> DynTrustIter<'a> {
    #[inline]
    pub fn vsum(self) -> TResult<Scalar> {
        match_trust_iter!(self; Numeric(e) => Ok(e.vsum().unwrap_or_default().into()),)
    }
}

// pub trait Node2: 'static {
//     fn as_any(&self) -> &dyn std::any::Any;
// }

// pub trait TypeNode<I, O> {
//     fn eval<InTyp: IntoIterator<Item = I>>(&self, data: InTyp) -> TResult<O>;
// }

// // trait Numeric: IsNone + Zero {}
// // impl Numeric for f64 {}
// // impl Numeric for f32 {}

// pub struct VSumNode {}

// impl Node2 for VSumNode {
//     #[inline]
//     fn as_any(&self) -> &dyn std::any::Any {
//         self
//     }
// }
// impl Node2 for Box<dyn Node2> {
//     #[inline]
//     fn as_any(&self) -> &dyn std::any::Any {
//         self.as_ref().as_any()
//     }
// }

// impl<I: IsNone> TypeNode<I, I> for VSumNode
// where
//     I::Inner: Zero + Default,
// {
//     fn eval<InTyp: IntoIterator<Item = I>>(&self, data: InTyp) -> TResult<I> {
//         Ok(I::from_inner(data.vsum().unwrap_or_default()))
//     }
// }

// impl<N: TypeNode<I, O>, I, O> TypeNode<I, O> for Box<N> {
//     fn eval<InTyp: IntoIterator<Item = I>>(&self, data: InTyp) -> TResult<O> {
//         self.as_ref().eval(data)
//     }
// }

// struct Expr {
//     pub nodes: Vec<Box<dyn Node2>>,
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     #[test]
//     fn test_one_node() {
//         let expr = Expr {
//             nodes: vec![Box::new(VSumNode {})],
//         };
//         let data = vec![1.0, 2.0, 3.0];
//     }
// }
