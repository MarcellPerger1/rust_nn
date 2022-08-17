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

    fn request_nudge(&self, _nudge: f64) {}
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
    pub bias_nudge_sum: RefCell<f64>,
    pub inp_w_nudge_sum: RefCell<Vec<f64>>,
    pub nudge_cnt: RefCell<i32>,
    pub requested_nudge: RefCell<f64>,
}
impl_as_any!(Node);
impl Node {
    pub fn new(bias: f64, inp_w: &Vec<f64>, layer: usize) -> Node {
        return Self {
            bias,
            inp_w: inp_w.clone(),
            layer,
            result_cache: RefCell::new(None),
            sum_cache: RefCell::new(None),
            bias_nudge_sum: RefCell::new(0.0),
            inp_w_nudge_sum: RefCell::new(vec![0.0; inp_w.len()]),
            nudge_cnt: RefCell::new(0),
            requested_nudge: RefCell::new(0.0),
        };
    }
    pub fn invalidate(&self) {
        self.sum_cache.replace(None);
        self.result_cache.replace(None);
    }

    pub fn is_last_layer(&self, network: &Network) -> bool {
        self.layer == network.n_layers() - 1
    }

    pub fn get_weight(&self, wi: usize) -> f64 {
        self.inp_w[wi]
    }
    pub fn set_weight(&mut self, wi: usize, v: f64) {
        self.inp_w[wi] = v;
    }

    pub fn calc_nudge(&self, network: &Network) {
        let d_sig = self.get_sum(network);
        let base_nudge = *self.requested_nudge.borrow() * d_sig * network.config.learning_rate;
        // bias nudge
        *self.bias_nudge_sum.borrow_mut() += base_nudge;
        (0..self.inp_w.len()).for_each(|i| {
            // weight nudges
            (*self.inp_w_nudge_sum.borrow_mut())[i] +=
                base_nudge * network.get_node(self.layer - 1, i).get_value(network);
            // nudge previous nodes
            network
                .get_node(self.layer - 1, i)
                .request_nudge(base_nudge * self.inp_w[i])
        });
        (*self.nudge_cnt.borrow_mut()) += 1;
    }

    pub fn request_nudge(&self, nudge: f64) {
        *self.requested_nudge.borrow_mut() += nudge;
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
}
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
