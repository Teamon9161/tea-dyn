use crate::prelude::*;
use derive_more::{From, IsVariant};
#[cfg(feature = "pl")]
use polars::prelude::Series;

#[derive(From, Clone, Debug, IsVariant)]
pub enum Data<'a> {
    TrustIter(Arc<DynTrustIter<'a>>),
    Scalar(Arc<Scalar>),
    Vec(Arc<DynVec>),
    Array(Arc<DynArray<'a>>),
    #[cfg(feature = "pl")]
    Series(Series),
}

impl<'a> From<DynTrustIter<'a>> for Data<'a> {
    #[inline]
    fn from(iter: DynTrustIter<'a>) -> Self {
        Data::TrustIter(Arc::new(iter))
    }
}

impl<'a> From<DynArray<'a>> for Data<'a> {
    #[inline]
    fn from(arr: DynArray<'a>) -> Self {
        Data::Array(Arc::new(arr))
    }
}

impl<T: GetDataType> From<Vec<T>> for Data<'_> {
    #[inline]
    fn from(vec: Vec<T>) -> Self {
        let vec: DynVec = vec.into();
        vec.into()
    }
}

impl From<DynVec> for Data<'_> {
    #[inline]
    fn from(vec: DynVec) -> Self {
        Data::Vec(Arc::new(vec))
    }
}

trait ScalarElement {}

impl ScalarElement for bool {}
impl ScalarElement for f32 {}
impl ScalarElement for f64 {}
impl ScalarElement for i32 {}
impl ScalarElement for i64 {}
impl ScalarElement for u8 {}
impl ScalarElement for u64 {}
impl ScalarElement for usize {}
impl ScalarElement for String {}
impl ScalarElement for Option<usize> {}
#[cfg(feature = "time")]
impl ScalarElement for DateTime {}
#[cfg(feature = "time")]
impl ScalarElement for TimeDelta {}

impl<T: ScalarElement> From<T> for Data<'_>
where
    T: Into<Scalar>,
{
    #[inline]
    fn from(v: T) -> Self {
        let s: Scalar = v.into();
        s.into()
    }
}

impl From<Scalar> for Data<'_> {
    #[inline]
    fn from(vec: Scalar) -> Self {
        Data::Scalar(Arc::new(vec))
    }
}

impl<'a> Data<'a> {
    #[inline]
    pub fn backend(&self) -> Backend {
        match self {
            Data::TrustIter(_) | Data::Scalar(_) | Data::Vec(_) => Backend::Vec,
            Data::Array(_) => Backend::Numpy,
            #[cfg(feature = "pl")]
            Data::Series(_) => Backend::Polars,
        }
    }

    #[inline]
    pub fn alias(self, name: Option<&str>) -> Self {
        // currently only Series has name
        #[cfg(feature = "pl")]
        if let Some(name) = name {
            if let Data::Series(series) = self {
                return Data::Series(series.with_name(name));
            }
        }
        self
    }

    #[inline]
    /// if output of the expression is a trust iter, consume and convert it to a result
    pub fn into_result(self, backend: Option<Backend>) -> TResult<Self> {
        if let Data::TrustIter(iter) = self {
            // this function only works for trust iter
            let backend = backend.unwrap_or(Backend::Numpy);
            Arc::try_unwrap(iter).unwrap().collect(backend)
        } else {
            Ok(self)
        }
    }

    #[inline]
    pub fn into_scalar(self) -> TResult<Scalar> {
        self.try_into_scalar()
            .map_err(|_| terr!("Data cannot be converted into scalar"))
    }

    #[inline]
    #[allow(clippy::should_implement_trait)]
    pub fn into_titer(self) -> TResult<DynTrustIter<'a>> {
        self.try_into_iter()
            .map_err(|_| terr!("Data cannot be converted into iterator"))
    }

    #[inline]
    #[allow(clippy::missing_transmute_annotations)]
    pub fn into_array(self) -> TResult<DynArray<'a>> {
        if let Data::Array(array) = self {
            match Arc::try_unwrap(array) {
                Ok(array) => Ok(array),
                Err(array) => {
                    // the data is still shared
                    // this should only happen when the data is stored in a context
                    // so it is safe to reference data
                    Ok(unsafe { std::mem::transmute(array.view()) })
                }
            }
        } else {
            let vec = self
                .into_vec()
                .map_err(|_| terr!("Can not convert data to an array"))?;
            DynArray::from_vec(vec)
        }
    }

    #[inline]
    #[cfg(feature = "pl")]
    #[allow(clippy::missing_transmute_annotations)]
    pub fn into_series(self) -> TResult<Series> {
        match self {
            Data::Series(series) => Ok(series),
            Data::TrustIter(iter) => Arc::try_unwrap(iter).unwrap().collect_series(),
            _ => tbail!("Can not convert data to series, not implemented yet"),
        }
    }

    #[inline]
    pub fn into_vec(self) -> TResult<DynVec> {
        match self {
            Data::Vec(vec) => match Arc::try_unwrap(vec) {
                Ok(vec) => Ok(vec),
                Err(_) => {
                    tbail!("Can not convert data into vector as it is shared")
                }
            },
            Data::TrustIter(iter) => {
                if let Ok(iter) = Arc::try_unwrap(iter) {
                    iter.collect_vec()
                } else {
                    tbail!("Can not convert iterator into vector as it is shared")
                }
            }
            Data::Array(array) => {
                if let Ok(array) = Arc::try_unwrap(array) {
                    array.into_vec()
                } else {
                    tbail!("Can not convert array into vector as it is shared")
                }
            }
            // TODO: should we convert scalar to vec?
            _ => tbail!("Data is not a vector"),
        }
    }

    #[allow(unreachable_patterns)]
    pub fn try_into_scalar(self) -> Result<Scalar, Self> {
        match self {
            Data::TrustIter(iter) => {
                let iter = Arc::try_unwrap(iter).map_err(Data::TrustIter)?;
                let out: Scalar = match_trust_iter!(iter; Dynamic(i) => {
                    let vec = i.collect_trusted_to_vec();
                    if vec.len() == 1 {
                        Ok(vec.into_iter().next().unwrap().into())
                    } else {
                        return Err(vec.into())
                    }
                },)
                .unwrap();
                Ok(out)
            }
            Data::Scalar(scalar) => match Arc::try_unwrap(scalar) {
                Ok(s) => Ok(s),
                Err(s) => s.cheap_clone().ok_or_else(|| Data::Scalar(s)),
            },
            Data::Vec(vec) => {
                if vec.len() == 1 {
                    Ok(vec.get(0).unwrap())
                } else {
                    Err(vec.into())
                }
            }
            Data::Array(array) => {
                if array.len() == 1 {
                    Ok(array.get(0).unwrap())
                } else {
                    Err(array.into())
                }
            }
            #[cfg(feature = "pl")]
            Data::Series(s) => {
                if s.len() == 1 {
                    Ok(s.get(0)
                        .unwrap()
                        .try_into()
                        .expect("can not convert polars anyvalue to scalar"))
                } else {
                    Err(Data::Series(s))
                }
            }
        }
    }

    /// try consume the data and convert it into an iterator
    /// note that consume polars series and turn it into an iterator is not supported
    pub fn try_into_iter(self) -> Result<DynTrustIter<'a>, Self> {
        match self {
            Data::TrustIter(iter) => Arc::try_unwrap(iter).map_err(|iter| iter.into()),
            Data::Vec(vec) => {
                match Arc::try_unwrap(vec) {
                    Ok(vec) => Ok(vec.into_titer().unwrap()),
                    Err(vec) => {
                        // the data is still shared
                        // this should only happen when the data is stored in a context
                        // so it is safe to reference data
                        let iter: DynTrustIter<'a> =
                            unsafe { std::mem::transmute(vec.titer().unwrap()) };
                        Ok(iter)
                    }
                }
            }
            Data::Scalar(scalar) => {
                match Arc::try_unwrap(scalar) {
                    Ok(scalar) => Ok(scalar.into_titer().unwrap()),
                    Err(scalar) => {
                        // the data is still shared
                        // this should only happen when the data is stored in a context
                        // so it is safe to reference data
                        let iter: DynTrustIter<'a> =
                            unsafe { std::mem::transmute(scalar.titer().unwrap()) };
                        Ok(iter)
                    }
                }
            }
            // #[cfg(feature = "ndarray")]
            Data::Array(array) => {
                if array.ndim() <= 1 {
                    match Arc::try_unwrap(array) {
                        Ok(array) => Ok(array.into_titer().unwrap()),
                        Err(array) => {
                            // the data is still shared
                            // this case should only happen when the data is stored in a context
                            // so it is safe to reference data
                            let iter: DynTrustIter<'a> =
                                unsafe { std::mem::transmute(array.titer().unwrap()) };
                            Ok(iter)
                        }
                    }
                } else {
                    Err(array.into())
                }
            }
            #[cfg(feature = "pl")]
            Data::Series(s) => Err(Data::Series(s)),
        }
    }

    /// try get an iterator from the data, but we cannn't create an iterator if
    /// data is a TrustIter
    pub fn try_titer(&self) -> TResult<DynTrustIter> {
        match self {
            Data::TrustIter(_iter) => {
                // if let Some(iter) = Arc::get_mut(iter) {
                //     let iter = std::mem::take(iter);
                //     Ok(iter)
                // } else {
                //     tbail!("Can not iterate over a reference of iterator")
                // }
                tbail!("Can not iterate over a reference of iterator")
            }
            Data::Vec(vec) => vec.titer(),
            Data::Scalar(scalar) => scalar.titer(),
            Data::Array(array) => array.titer(),
            #[cfg(feature = "pl")]
            Data::Series(s) => s.titer(),
        }
    }
}
