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
