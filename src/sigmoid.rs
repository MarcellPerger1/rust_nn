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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::*;

    #[test]
    fn sig_f32() {
        assert_f_eq!(0.0_f32.sigmoid(), 0.5_f32);
        assert_f_eq!(1.5_f32.sigmoid(), 0.8175744_f32);
    }
    #[test]
    fn sig_f64() {
        assert_f_eq!(0.0_f64.sigmoid(), 0.5_f64);
        assert_f_eq!(1.5_f64.sigmoid(), 0.8175744761936437_f64);
    }
    #[test]
    fn sig_deriv_f32() {
        assert_f_eq!(0.0_f32.sig_deriv(), 0.25_f32);
        assert_f_eq!((-1.8_f32).sig_deriv(), 0.121729344_f32);
    }
    #[test]
    fn sig_deriv_f64() {
        assert_f_eq!(0.0_f64.sig_deriv(), 0.25_f64);
        assert_f_eq!((-1.8_f64).sig_deriv(), 0.12172934028708539_f64);
    }
    // this is more of a test for my util funcs than of functionality
    for_f_types!(sig_inf, {
        assert_f_eq!(FT::INFINITY.sigmoid(), 1.0);
        assert_f_eq!(FT::NEG_INFINITY.sigmoid(), 0.0);
    });
    for_f_types!(sig_deriv_inf, {
        assert_f_eq!(FT::INFINITY.sig_deriv(), 0.0);
        assert_f_eq!(FT::NEG_INFINITY.sig_deriv(), 0.0);
    });
}
