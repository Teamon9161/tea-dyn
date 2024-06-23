use crate::prelude::*;

#[derive(Clone, Default)]
pub struct Expr {
    pub name: Option<Arc<str>>,
    pub nodes: Vec<Node>,
}

unsafe impl Send for Expr {}
unsafe impl Sync for Expr {}

#[inline]
pub fn s<S: Into<Symbol>>(symbol: S) -> Expr {
    let node: SelectNode = Into::<SelectNode>::into(symbol);
    Expr::new(node)
}

#[inline]
pub fn lit<V: Into<Scalar>>(v: V) -> Expr {
    let node = LitNode {
        value: Arc::new(v.into()),
    };
    Expr::new(node)
}

// pub type ExprFunc = Arc<dyn for<'b> Fn(&Context<'b>, Option<Backend>) -> TResult<Data<'b>>>;

impl Expr {
    #[inline]
    pub fn new<N: Into<Node>>(node: N) -> Self {
        let expr = Expr {
            name: None,
            nodes: vec![],
        };
        expr.chain(node)
    }

    #[inline]
    pub fn rename(&mut self, name: &str) -> &mut Self {
        self.name = Some(Arc::from(name));
        self
    }

    #[inline]
    pub fn alias(mut self, name: &str) -> Self {
        self.rename(name);
        self
    }

    #[inline]
    pub fn chain<N: Into<Node>>(mut self, node: N) -> Self {
        let node = node.into();
        if let Node::Select(n) = &node {
            if let Some(name) = n.name() {
                self.rename(name);
            }
        }
        self.nodes.push(node);
        self
    }

    pub fn to_func(
        &self,
    ) -> impl for<'b> Fn(&Context<'b>, Option<Backend>) -> TResult<Data<'b>> + '_ {
        move |ctx: &Context, backend: Option<Backend>| {
            let mut data: Option<Data> = None;
            let backend = backend.unwrap_or_default();
            // backend is the same for all nodes
            for node in &self.nodes {
                match node {
                    Node::Select(n) => {
                        data = Some(n.select(ctx)?);
                    }
                    Node::Lit(n) => {
                        data = Some(n.eval()?);
                    }
                    Node::Base(n) => {
                        data = Some((n.func)(
                            data.ok_or_else(|| terr!("Should select something to map as first"))?,
                            backend,
                        )?);
                    }
                    Node::Context(n) => {
                        data = Some((n.func)(
                            data.ok_or_else(|| terr!("Should select something to map as first"))?,
                            ctx,
                            backend,
                        )?);
                    }
                }
            }
            data.ok_or_else(move || terr!("No data to return"))
        }
        // Arc::new(func)
    }

    // pub fn to_func(
    //     &self,
    // ) -> impl for<'b> Fn(&Context<'b>, Option<Backend>) -> TResult<Data<'b>> + '_ {
    //     // let mut func = |(data, ctx, backend)| Ok((data, ctx, backend));
    //     let mut func: Box<
    //         dyn for<'b, 'c> Fn(
    //                 (&'c Context<'b>, Option<Backend>),
    //             )
    //                 -> TResult<(Option<Data<'b>>, &'c Context<'b>, Backend)>
    //             + '_,
    //     > = Box::new(|(ctx, backend)| {
    //         Ok((
    //             None,
    //             ctx,
    //             backend.unwrap_or(ctx.backend.unwrap_or(Backend::Vec)),
    //         ))
    //     });
    //     for node in &self.nodes {
    //         match node {
    //             Node::Select(n) => {
    //                 func = Box::new(move |(ctx, backend)| {
    //                     let (_, ctx, backend) = func((ctx, backend))?;
    //                     Ok((Some(n.select(ctx)?), ctx, backend))
    //                 });
    //             }
    //             Node::Lit(n) => {
    //                 func = Box::new(move |(ctx, backend)| {
    //                     let (_, ctx, backend) = func((ctx, backend))?;
    //                     Ok((Some(n.eval()?), ctx, backend))
    //                 });
    //                 // func = |(data, ctx, backend)| Ok((n.eval()?, ctx, backend));
    //             }
    //             Node::Base(n) => {
    //                 func = Box::new(move |(ctx, backend)| {
    //                     let (data, ctx, backend) = func((ctx, backend))?;
    //                     Ok((
    //                         Some((n.func)(
    //                             data.ok_or_else(|| {
    //                                 terr!("Should select something to map as first")
    //                             })?,
    //                             backend,
    //                         )?),
    //                         ctx,
    //                         backend,
    //                     ))
    //                 });
    //                 // func = |(data, ctx, backend)| Ok(((n.func)(data, backend)?, ctx, backend));
    //             }
    //             Node::Context(n) => {
    //                 func = Box::new(move |(ctx, backend)| {
    //                     let (data, ctx, backend) = func((ctx, backend))?;
    //                     Ok((
    //                         Some((n.func)(
    //                             data.ok_or_else(|| {
    //                                 terr!("Should select something to map as first")
    //                             })?,
    //                             ctx,
    //                             backend,
    //                         )?),
    //                         ctx,
    //                         backend,
    //                     ))
    //                 });
    //                 // func = |(data, ctx, backend)| Ok(((n.func)(data, ctx, backend)?, ctx, backend));
    //             }
    //         }
    //     }
    //     return move |ctx, backend| {
    //         let (data, _ctx, _backend) = func((ctx, backend))?;
    //         data.ok_or_else(|| terr!("No data to return"))
    //     };
    // }

    #[inline]
    pub fn eval<'a, 'b>(
        &'a self,
        ctx: &'a Context<'b>,
        backend: Option<Backend>,
    ) -> TResult<Data<'b>> {
        let name = self.name.as_ref().map(AsRef::as_ref);
        let func = self.to_func();
        let backend = if backend.is_none() && ctx.backend.is_some() {
            ctx.backend
        } else {
            backend
        };
        Ok(func(ctx, backend)?.into_result(backend)?.alias(name))
    }
}
