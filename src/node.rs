use crate::network::Network;
use crate::sigmoid::Sigmoid;
use crate::util::{impl_as_any, AsAny, TryIntoRef, TryIntoRefMut};
use std::cell::RefCell;

pub type AnyNode = Box<dyn NodeLike>;

pub trait NodeLike: AsAny + std::fmt::Debug {
    fn get_value(&self, network: &Network) -> f64;

    fn invalidate(&self) {
        // do nothing by default
    }
}

impl TryIntoRef for dyn NodeLike {
    fn try_into_ref<T: 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref()
    }
}
impl TryIntoRefMut for dyn NodeLike {
    fn try_into_ref_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.as_any_mut().downcast_mut()
    }
}

pub fn new_node<T: NodeLike + 'static>(n: T) -> AnyNode {
    Box::new(n) as AnyNode
}

#[derive(Debug)]
pub struct Node {
    pub bias: f64,
    pub inp_w: Vec<f64>,
    pub layer: usize,
    pub(crate) sum_cache: RefCell<Option<f64>>,
    pub(crate) result_cache: RefCell<Option<f64>>,
    pub bias_nudge_sum: f64,
    pub inp_w_nudge_sum: Vec<f64>,
    pub nudge_cnt: i32,
    pub requested_nudge: f64,
}
impl Node {
    pub fn new(bias: f64, inp_w: &Vec<f64>, layer: usize) -> Node {
        let inp_w = inp_w.clone();
        return Self {
            bias,
            inp_w,
            layer,
            result_cache: RefCell::new(None),
            sum_cache: RefCell::new(None),
            bias_nudge_sum: 0.0,
            inp_w_nudge_sum: Vec::new(),
            nudge_cnt: 0,
            requested_nudge: 0.0,
        };
    }
    pub fn invalidate(&self) {
        self.sum_cache.replace(None);
        self.result_cache.replace(None);
    }

    pub fn is_last_layer(&self, network: &Network) -> bool {
        self.layer == network.shape.len() - 1
    }

    pub fn get_weight(&self, wi: usize) -> f64 {
        self.inp_w[wi]
    }
    pub fn set_weight(&mut self, wi: usize, v: f64) {
        self.inp_w[wi] = v;
    }

    pub fn get_sum(&self, network: &Network) -> f64 {
        if let Some(cached) = *self.result_cache.borrow() {
            return cached;
        }
        let inp_sum = self
            .inp_w
            .iter()
            .enumerate()
            .map(|(i, v)| network.get_node(self.layer - 1, i).get_value(&network) * v)
            .sum::<f64>()
            + self.bias;
        self.sum_cache.replace(Some(inp_sum));
        inp_sum
    }

    pub(crate) fn request_nudge(&mut self, nudge: f64) {
        self.requested_nudge += nudge;
    }
}
impl_as_any!(Node);
impl NodeLike for Node {
    // have to pass it in because circular data structures in Rust never end well
    // (that time i tried to implement a (doubly) linked list using only safe Rust,
    // it did not go well)
    fn get_value(&self, network: &Network) -> f64 {
        if let Some(cached) = *self.result_cache.borrow() {
            return cached;
        }
        let inp_sum = self.get_sum(network);
        let val = inp_sum.sigmoid();
        self.result_cache.replace(Some(val));
        return val;
    }
}

#[derive(Debug)]
pub struct StartNode {
    pub(crate) value: f64,
}
impl_as_any!(StartNode);
impl NodeLike for StartNode {
    fn get_value(&self, _: &Network) -> f64 {
        self.value
    }
}
impl StartNode {
    pub fn new(value: f64) -> StartNode {
        StartNode { value }
    }

    pub fn set_value(&mut self, value: f64) {
        self.value = value;
    }
}
