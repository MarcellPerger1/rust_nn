pub(crate) trait GetEpsilon {
    fn epsilon(&self) -> Self;
}
macro_rules! epsilon_impl {
    ( $( $t:ty ),+ ) => {
        $(
            impl GetEpsilon for $t {
                fn epsilon(&self) -> Self {
                    <$t>::EPSILON
                }
            }
        )+
    };
}
epsilon_impl!(f32, f64);

macro_rules! assert_f_eq {
    (@to_bool $a:expr, $b:expr) => {
        {
            let __a = $a;
            let __b = $b;
            // NaN != NaN workaround
            let __a_nan = __a != __a;
            let __b_nan = __b != __b;
            if __a_nan || __b_nan { __a_nan && __b_nan }
            else {
                __a == __b || ((__a - __b).abs() / __b <= (__b.epsilon() * 4.0))
            }
        }
    };
    ($a:expr, $b:expr $(, $rest:tt )?) => {
        {
            let __a = $a;
            let __b = $b;
            if(!assert_f_eq!(@to_bool __a, __b)){
                assert_eq!(__a, __b $($rest)?);
            }
        }
    }
}
pub(crate) use assert_f_eq;

macro_rules! assert_refcell_eq {
    ($refcell: expr, $value: expr) => {
        assert_eq!(*$refcell.borrow(), $value)
    };
}
pub(crate) use assert_refcell_eq;

macro_rules! for_f_types {
    ($name:ident, $test:block) => {
        #[allow(unused_imports)]
        mod $name {
            use super::*;
            #[test]
            fn test_f32() {
                type FT = f32;
                $test
            }
            #[test]
            fn test_f64() {
                type FT = f64;
                $test
            }
        }
    };
}
pub(crate) use for_f_types;
