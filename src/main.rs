pub mod sigmoid;
pub mod network;
mod util;

use std::cell::RefCell;
use crate::sigmoid::Sigmoid;
use crate::network::Network;


fn error_f(a: f64, b: f64) -> f64 {
    (a - b).powi(2) // cleaner and (with optimisations) probably just as fast as *
}

// region Node
#[derive(Debug)]
pub enum AnyNode {
    Start(StartNode),
    Normal(Node),
}
impl AnyNode {
    pub fn get_value(&self, n: &Network) -> f64 {
        match self {
            Self::Start(v) => v.get_value(n),
            Self::Normal(v) => v.get_value(n),
        }
    }
}

pub trait NodeValue {
    fn get_value(&self, network: &Network) -> f64;
}

#[derive(Debug)]
pub struct Node {
    pub bias: f64,
    pub inp_w: Vec<f64>,
    pub layer: usize,
    sum_cache: RefCell<Option<f64>>,
    result_cache: RefCell<Option<f64>>,
    pub bias_nudge_sum: f64,
    pub inp_w_nudge_sum: Vec<f64>,
    pub nudge_cnt: i32,
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
        };
    }
    pub fn invalidate(&self) {
        self.sum_cache.replace(None);
        self.result_cache.replace(None);
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
}

impl NodeValue for Node {
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
    value: f64,
}
impl NodeValue for StartNode {
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
//endregion

macro_rules! assert_cached_eq {
    ($network:expr, $ni:expr, $li:expr, $val:expr) => {
        assert_eq!(
            *$network.get_main_node($ni, $li).result_cache.borrow(),
            $val
        )
    };
    ($cache:expr, $val:expr) => {
        assert_eq!(*$cache.borrow(), $val)
    };
}

fn run_checks() {
    // NOTE TO SELF: remember that activation is passed thru sigmoid activation
    // DONT FORGET THIS when writing tests and wondering why they fail
    let mut nw = Network::new(&vec![2, 2]);
    // println!("{:#?}", nw);
    assert_cached_eq!(nw, 1, 1, None);
    let v = nw.get_node(1, 1).get_value(&nw);
    assert_eq!(v, 0.5); // (0.0).sigmoid()
    assert_cached_eq!(nw, 1, 1, Some(v));
    nw.get_start_node_mut(1).set_value(1.0);
    assert_eq!(nw.get_start_node(1).value, 1.0);
    nw.get_main_node_mut(1, 1).set_weight(1, 1.0);
    // NOTE: the reason its commented out is
    // because i might want set_weight to invalidate cache
    // assert_eq!(*nw.get_main_node(1, 1).result_cache.borrow(), Some(v));
    assert_eq!(nw.get_main_node(1, 1).get_weight(1), 1.0);
    nw.get_main_node(1, 1).invalidate();
    assert_cached_eq!(nw, 1, 1, None);
    let v = nw.get_main_node(1, 1).get_value(&nw);
    assert_eq!(v, 0.7310585786300049);
    assert_cached_eq!(nw, 1, 1, Some(v));
    assert_cached_eq!(nw.get_main_node(1, 1).sum_cache, Some(1.));
    assert_eq!(nw.get_current_cost(&vec![0.5, 1.0]), 0.07232948812851325);
    println!("{:#?}", nw);
}

fn main() {
    println!("Hello world!");
    run_checks();
}
