use crate::prelude::*;
use pyo3::prelude::*;

#[pymethods]
impl PyExpr {
    #[pyo3(signature=(nan=true))]
    pub fn abs(&self, nan: bool) -> Self {
        if nan {
            self.clone().0.vabs().into()
        } else {
            self.clone().0.abs().into()
        }
    }

    #[pyo3(signature=(n, value=None, axis=None, par=None))]
    pub fn shift(
        &self,
        n: &Bound<'_, PyAny>,
        value: Option<&Bound<'_, PyAny>>,
        axis: Option<usize>,
        par: Option<bool>,
    ) -> PyResult<Self> {
        let n: PyExpr = n.try_into()?;
        let value: Option<PyExpr> = value.map(|v| v.try_into().unwrap());
        let n = n.0.clone();
        let value = value.map(|v| v.0.clone());
        Ok(self.clone().0.vshift(n, value, axis, par).into())
    }
}
