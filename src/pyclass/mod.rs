use crate::prelude::*;
use derive_more::Deref;
use pyo3::{exceptions::PyValueError, prelude::*};
mod from_py;
mod into_py;
mod methods;

// #[pyclass(name = "Context")]
// pub struct PyContext(Context<'static>);

#[pyclass(name = "Expr")]
#[derive(Clone, Deref)]
pub struct PyExpr(Expr);

impl TryFrom<&Bound<'_, PyAny>> for PyExpr {
    type Error = PyErr;
    #[inline]
    fn try_from(b: &Bound<'_, PyAny>) -> PyResult<PyExpr> {
        if let Ok(expr) = b.extract::<PyExpr>() {
            Ok(expr)
        } else if let Ok(scalar) = b.extract::<Scalar>() {
            Ok(lit(scalar).into())
        } else {
            Err(PyValueError::new_err(format!(
                "can not convert {:?} to Expr",
                b
            )))
        }
    }
}

impl From<Expr> for PyExpr {
    #[inline]
    fn from(expr: Expr) -> Self {
        Self(expr)
    }
}

#[pyfunction(name = "lit")]
pub fn py_lit(v: Bound<'_, PyAny>) -> PyResult<PyExpr> {
    let v: Scalar = v.extract()?;
    Ok(lit(v).into())
}

#[pyfunction(name = "s")]
pub fn py_s(i: Bound<'_, PyAny>) -> PyResult<PyExpr> {
    let i: i32 = i.extract()?;
    Ok(s(i).into())
}

#[pymethods]
impl PyExpr {
    #[new]
    pub fn new() -> PyResult<Self> {
        let expr = Expr::default();
        Ok(expr.into())
    }

    pub fn alias(&self, name: &str) -> Self {
        self.clone().0.alias(name).into()
    }

    #[pyo3(signature=(ctx=None))]
    pub fn eval<'py>(
        &'py self,
        ctx: Option<&'py Bound<'py, PyAny>>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let ctx: Context<'py> = ctx.map(|c| c.extract().unwrap()).unwrap_or_default();
        let out = self
            .0
            .eval(&ctx)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(out.into_py(py))
    }
}
