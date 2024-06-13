use crate::prelude::*;
use derive_more::Deref;
use pyo3::{exceptions::PyValueError, prelude::*};

#[pyfunction(name = "lit")]
pub fn py_lit(v: Bound<'_, PyAny>) -> PyResult<PyExpr> {
    let v: Scalar = v.extract()?;
    Ok(lit(v).into())
}

#[pyfunction(name = "s")]
pub fn py_s(i: Bound<'_, PyAny>) -> PyResult<PyExpr> {
    let i: Symbol = i.extract()?;
    Ok(s(i).into())
}

#[pyclass(name = "Expr")]
#[repr(transparent)]
#[derive(Clone, Deref)]
pub struct PyExpr(pub Expr);

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

    #[pyo3(signature=(ctx=None, backend=None))]
    pub fn eval<'py>(
        &'py self,
        ctx: Option<Bound<'py, PyAny>>,
        backend: Option<&'py Bound<'py, PyAny>>,
        py: Python<'py>,
    ) -> PyResult<PyObject> {
        let ctx_obj = ctx.clone();
        let ctx: Context<'py> = ctx.map(|c| c.extract().unwrap()).unwrap_or_default();
        let backend: Option<Backend> = backend.map(|bk| bk.extract().unwrap());
        let out = self
            .0
            .eval(&ctx, backend)
            .map_err(|e| PyValueError::new_err(format!("{}", e)))?;
        out.try_into_py(py, ctx_obj)
    }
}
