use std::cell::RefCell;

pub trait Sigmoid {
    fn sigmoid(&self) -> Self;
}
impl Sigmoid for f32 {
    fn sigmoid(&self) -> Self {
        1.0f32 / (1.0f32 + (-self).exp())
    }
}
impl Sigmoid for f64 {
    fn sigmoid(&self) -> Self {
        1. / (1. + (-self).exp())
    }
}

pub trait NodeValue {
    fn get_value(&self) -> f64;
}

pub struct Node {
    pub bias: f64,
    pub inp_w: Vec<f64>,
    prev_v: Vec<f64>,
    result_cache: RefCell<Option<f64>>,
}
impl Node{
    fn invalidate(&self){
        self.result_cache.replace(None);
    }
}

impl NodeValue for Node {
    fn get_value(&self) -> f64 {
        if let Some(cached) = *self.result_cache.borrow(){
            return cached;
        }
        let val = ((&self.inp_w)
            .iter()
            .enumerate()
            .map(|t| self.prev_v[t.0] * t.1)
            .sum::<f64>()
            + self.bias).sigmoid();
        self.result_cache.replace(Some(val));
        return val;
    }
}

pub struct StartNode {
    value: f64,
}
impl NodeValue for StartNode {
    fn get_value(&self) -> f64 {
        self.value
    }
}

fn main() {
    println!("Hello world!");
    let n = Node {
        bias: 0.2,
        inp_w: vec![1.0, 2.0],
        prev_v: vec![0.7, 0.6],
        result_cache: RefCell::new(None),
    };
    let expect_v = 0.8909031788043871;
    let v = n.get_value();
    println!("v={}", v);
    assert_eq!(v, expect_v);
    println!("cached={:?}", n.result_cache);
    assert_eq!(*n.result_cache.borrow(), Some(expect_v));
    let v = n.get_value();
    println!("call-again={}", v);
    assert_eq!(v, expect_v);
    n.invalidate();
    assert_eq!(*n.result_cache.borrow(), None);
    println!("cached={:?}", n.result_cache);
}
