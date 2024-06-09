mod methods;
mod structs;
#[macro_use]
mod macros;

pub use structs::*;

use tevec::prelude::DataType;

pub trait GetDtype {
    fn dtype(&self) -> DataType;
}
