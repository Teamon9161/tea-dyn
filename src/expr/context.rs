use super::data::Data;

#[derive(Debug, Clone, Copy)]
pub enum Backend {
    Pandas,
    // Polars,
    Numpy,
    Vec,
}

#[derive(Default)]
pub struct Context<'a> {
    pub data: Vec<Data<'a>>,
    pub backend: Option<Backend>,
}

impl<'a> Context<'a> {
    #[inline]
    pub fn new<D: Into<Data<'a>>>(d: D) -> Self {
        Context {
            data: vec![d.into()],
            backend: None,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}
