pub trait Sigmoid {
    fn sigmoid(&self) -> Self;
    fn sig_deriv(&self) -> Self;
}
impl Sigmoid for f32 {
    #[inline]
    fn sigmoid(&self) -> Self {
        1.0f32 / (1.0f32 + (-self).exp())
    }
    fn sig_deriv(&self) -> Self {
        let s = self.sigmoid();
        return s * (1.0f32 - s);
    }
}
impl Sigmoid for f64 {
    #[inline]
    fn sigmoid(&self) -> Self {
        1. / ((1 as Self) + (-self).exp())
    }
    fn sig_deriv(&self) -> Self {
        let s = self.sigmoid();
        return s * (1.0 - s);
    }
}