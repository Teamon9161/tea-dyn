use tevec::prelude::{tbail, tensure, terr, TResult};

use super::data::Data;
use derive_more::From;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc};

#[derive(Debug, Default, Clone, Copy)]
pub enum Backend {
    #[default]
    Vec,
    Numpy,
    Pandas,
    #[cfg(feature = "pl")]
    Polars,
}

/// Symbol is used to select data from context
#[derive(Clone, From)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]

pub enum Symbol {
    I32(i32),
    Usize(usize),
    Str(Arc<str>),
}

impl Symbol {
    #[inline]
    pub fn name(&self) -> Option<&str> {
        if let Symbol::Str(name) = self {
            Some(name)
        } else {
            None
        }
    }
}

impl From<&str> for Symbol {
    #[inline]
    fn from(value: &str) -> Self {
        Symbol::Str(value.into())
    }
}

impl From<String> for Symbol {
    #[inline]
    fn from(value: String) -> Self {
        Symbol::Str(value.into())
    }
}

// TODO: improve debug info of Context
#[derive(Default, Debug)]
pub struct Context<'a> {
    pub data: Vec<Data<'a>>,
    pub backend: Option<Backend>,
    pub col_map: Option<HashMap<Arc<str>, usize>>,
}

impl<'a> Context<'a> {
    #[inline]
    #[allow(clippy::vec_init_then_push)]
    pub fn new<D: Into<Data<'a>>>(d: D) -> Self {
        let mut data = Vec::with_capacity(1);
        data.push(d.into());
        Context {
            data,
            backend: None,
            col_map: None,
        }
    }

    #[inline]
    pub fn new_from_data_column<C: IntoIterator<Item = T>, T: Into<Arc<str>>>(
        data: Vec<Data<'a>>,
        columns: C,
    ) -> Self
    where
        C::IntoIter: ExactSizeIterator,
    {
        let columns = columns.into_iter();
        let mut col_map = HashMap::with_capacity(columns.len() * 2);
        for (i, col) in columns.into_iter().enumerate() {
            col_map.insert(col.into(), i);
        }
        Context {
            data,
            backend: None,
            col_map: Some(col_map),
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

    pub fn get<S: Into<Symbol>>(&self, symbol: S) -> TResult<&Data<'a>> {
        let symbol = symbol.into();
        match symbol {
            Symbol::Usize(idx) => self
                .data
                .get(idx)
                .ok_or_else(|| terr!("index {} out of bounds", idx)),
            Symbol::I32(idx) => {
                let idx = if idx < 0 {
                    self.len() as i32 + idx
                } else {
                    idx
                };
                tensure!(idx >= 0, "negative index is out of bounds");
                self.get(idx as usize)
            }
            Symbol::Str(name) => {
                if let Some(map) = &self.col_map {
                    let idx = map
                        .get(name.as_ref())
                        .ok_or_else(|| terr!("column {} not found in context", name))?;
                    self.get(*idx)
                } else {
                    tbail!("Cannot get column by name, the context does not have column map")
                }
            }
        }
    }
}
