#[macro_export]
macro_rules! expect_cast {
    ($var:expr => $t:path) => {
        if let $t(v) = $var {
            v
        } else {
            unreachable!()
        }
    };
    ($var:expr, $t:path) => {
        expect_cast!($var => $t)
    }
}

pub(crate) use expect_cast;


pub fn error_f(a: f64, b: f64) -> f64 {
    (a - b).powi(2) // cleaner and (with optimisations) probably just as fast as *
}
