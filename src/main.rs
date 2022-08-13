pub mod sigmoid;

use std::cell::RefCell;
use crate::sigmoid::Sigmoid;

fn error_f(a: f64, b: f64) -> f64 {
    (a - b).powi(2) // cleaner and (with optimisations) probably just as fast as *
}

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

pub type LayerT = Vec<AnyNode>;
pub type NetworkLayersT = Vec<LayerT>;
#[derive(Debug)]
pub struct Network {
    shape: Vec<usize>,
    layers: NetworkLayersT,
}

// main impl
impl Network {
    pub fn new(shape: &Vec<usize>) -> Network {
        let shape = shape.clone();
        let layers = shape
            .iter()
            .enumerate()
            .map(|(i, n)| {
                (0..*n)
                    .map(|_| {
                        // todo move this conditional out of the closure
                        if i != 0 {
                            AnyNode::Normal(Node::new(0., &vec![0.; shape[i - 1]], i))
                        } else {
                            AnyNode::Start(StartNode::new(0.))
                        }
                    })
                    .collect::<LayerT>()
            })
            .collect();
        let nw = Network { shape, layers };
        return nw;
    }

    pub fn get_current_cost(&self, expected: &Vec<f64>) -> f64 {
        assert_eq!(self.layers.len(), expected.len());
        self.layers[self.shape.len() - 1]
            .iter()
            .enumerate()
            .map(|(i, n)| {
                error_f(
                    expected[i],
                    expect_cast!(n=>AnyNode::Normal).get_value(&self),
                )
            })
            .sum()
    }

    pub fn invalidate(&self) {
        let mut li = self.layers.iter();
        li.next();
        for l in li {
            for n in l {
                expect_cast!(n => AnyNode::Normal).invalidate();
            }
        }
    }
}

// indexing stuff
impl Network {
    #[inline]
    pub fn layers_ref(&self) -> &NetworkLayersT {
        &self.layers
    }
    #[inline]
    pub fn layers_mut(&mut self) -> &mut NetworkLayersT {
        &mut self.layers
    }
    pub fn get_layer(&self, i: usize) -> &LayerT {
        &self.layers_ref()[i]
    }

    pub fn get_node(&self, li: usize, ni: usize) -> &AnyNode {
        &self.get_layer(li)[ni]
    }
    pub fn get_node_mut(&mut self, li: usize, ni: usize) -> &mut AnyNode {
        &mut self.layers[li][ni]
    }

    pub fn get_start_node(&self, ni: usize) -> &StartNode {
        expect_cast!(self.get_node(0, ni) => AnyNode::Start)
    }
    pub fn get_start_node_mut(&mut self, ni: usize) -> &mut StartNode {
        expect_cast!(self.get_node_mut(0, ni) => AnyNode::Start)
    }
    pub fn get_main_node(&self, li: usize, ni: usize) -> &Node {
        assert_ne!(li, 0, "no main nodes in layer 0");
        expect_cast!(self.get_node(li, ni) => AnyNode::Normal)
    }
    pub fn get_main_node_mut(&mut self, li: usize, ni: usize) -> &mut Node {
        assert_ne!(li, 0, "no main nodes in layer 0");
        expect_cast!(self.get_node_mut(li, ni) => AnyNode::Normal)
    }
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
