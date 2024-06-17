use crate::prelude::*;

#[derive(Clone, Default)]
pub struct Expr {
    pub name: Option<Arc<String>>,
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

impl Expr {
    #[inline]
    pub fn new<N: Into<Node>>(node: N) -> Self {
        Expr {
            name: None,
            nodes: vec![node.into()],
        }
    }

    #[inline]
    pub fn alias(mut self, name: &str) -> Self {
        self.name = Some(Arc::new(name.to_string()));
        self
    }

    #[inline]
    pub fn chain<N: Into<Node>>(mut self, node: N) -> Self {
        self.nodes.push(node.into());
        self
    }

    pub fn to_func<'a, 'b, 'c>(
        &'a self,
    ) -> Box<dyn Fn(&'c Context<'b>, Option<Backend>) -> TResult<Data<'b>> + 'a> {
        let func = |ctx, backend: Option<Backend>| {
            let mut data: Option<Data<'b>> = None;
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
            data.ok_or_else(|| terr!("No data to return"))
        };
        Box::new(func)
    }

    #[inline]
    pub fn eval<'a, 'b>(
        &'a self,
        ctx: &'a Context<'b>,
        backend: Option<Backend>,
    ) -> TResult<Data<'b>> {
        let func = self.to_func();
        let backend = if backend.is_none() && ctx.backend.is_some() {
            ctx.backend
        } else {
            backend
        };
        func(ctx, backend)?.into_result(backend)
    }
}
