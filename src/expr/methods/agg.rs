use crate::prelude::*;

impl Expr {
    pub fn sum(self) -> Expr {
        let node = BaseNode {
            name: "sum",
            func: Arc::new(|data, _backend| match data.try_into_iter() {
                Ok(iter) => Ok(iter.vsum()?.into()),
                Err(_data) => {
                    todo!()
                }
            }),
        };
        self.chain(node)
    }
}
