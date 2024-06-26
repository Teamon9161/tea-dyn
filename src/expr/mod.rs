mod context;
mod data;
mod expr_core;
mod methods;
mod node;

pub use context::{Backend, Context, Symbol};
pub use data::Data;
pub use expr_core::{lit, s, Expr};
pub use node::{BaseNode, CtxNode, LitNode, Node, SelectNode};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use tevec::prelude::*;
    #[test]
    #[cfg(feature = "map")]
    fn test_basic() -> TResult<()> {
        let ctx = Context {
            data: vec![
                d_vec![-1.0, 2.0, -3.0].into(),
                dt_iter![2.].into(),
                scalar!("hello".to_owned()).into(),
            ],
            backend: None,
            col_map: None,
        };
        // vec is still shared by context, so we will failed
        assert!(ctx.data[0].clone().into_vec().is_err());
        let res = s(0)
            .vabs()
            .abs()
            .eval(&ctx, Some(Backend::Vec))?
            .into_vec()?
            .f64()?;
        assert_eq!(res.as_ref(), &[1.0, 2.0, 3.0]);
        let res = s(2).eval(&ctx, None)?.into_scalar()?.string()?;
        assert_eq!(res.as_str(), "hello");
        // we cannot change context, so it should fail if the data is an iterator
        assert!(s(1).abs().vabs().eval(&ctx, None).is_err());
        Ok(())
    }

    #[test]
    fn test_select_by_str() -> TResult<()> {
        let ctx = Context::new_from_data_column(
            vec![
                d_vec![-1.0, 2.0, -3.0].into(),
                dt_iter![2.].into(),
                scalar!("hello".to_owned()).into(),
            ],
            ["a", "b", "c"],
        );
        assert!(s("c").eval(&ctx, None)?.into_scalar()?.string()?.as_str() == "hello");
        Ok(())
    }

    #[test]
    #[cfg(feature = "map")]
    fn test_ndarray_backend() -> TResult<()> {
        let ctx = Context::new(d1_array![-1.0, 2.0, -3.0]);
        let expr = s(0).abs().abs();
        let res = expr.eval(&ctx, Some(Backend::Numpy))?.into_vec()?.f64()?;
        assert_eq!(res.as_ref(), &[1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    #[cfg(feature = "map")]
    fn test_nd_expr() -> TResult<()> {
        use tevec::ndarray::prelude::*;
        let ctx = Context::new(d2_array![[1, -2, 3], [4, 5, -6], [-7, 8, 9]]);
        let expr = s(0).abs().vshift(lit(1), Some(lit(0)), None, None);
        let res = expr.eval(&ctx, None)?.into_array()?.i32()?;
        let expect: Array2<i32> = arr2(&[[0, 0, 0], [1, 2, 3], [4, 5, 6]]);
        assert_eq!(res.view().into_dimensionality().unwrap(), expect.view());
        Ok(())
    }
}
