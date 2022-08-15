pub fn error_f(a: f64, b: f64) -> f64 {
    (a - b).powi(2) // cleaner and (with optimisations) probably just as fast as *
}

pub trait AsAny {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

pub trait TryIntoRef {
    fn try_into_ref<T: 'static>(&self) -> Option<&T>;
}
pub trait TryIntoRefMut {
    fn try_into_ref_mut<T: 'static>(&mut self) -> Option<&mut T>;
}

// tod make a Derive() for this
#[macro_export]
macro_rules! impl_as_any {
    ($name:path) => {
        impl $crate::node::AsAny for $name {
            fn as_any(&self) -> &dyn ::std::any::Any {
                self
            }
            fn as_any_mut(&mut self) -> &mut dyn ::std::any::Any {
                self
            }
        }
    };
}

pub use impl_as_any;
