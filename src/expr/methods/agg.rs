use crate::prelude::*;

impl Expr {
    pub fn sum(self) -> Expr {
        let node = BaseNode {
            name: "sum",
            func: Arc::new(|data, _backend| match data.try_into_iter() {
                Ok(iter) => Ok(iter.vsum()?.into()),
                Err(data) => {
                    if let Ok(iter) = data.try_titer() {
                        Ok(iter.vsum()?.into())
                    } else {
                        todo!()
                    }
                }
            }),
        };
        self.chain(node)
    }
}
