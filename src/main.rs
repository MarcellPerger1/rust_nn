use std::cell::RefCell;

pub trait Sigmoid {
    fn sigmoid(&self) -> Self;
}
impl Sigmoid for f32 {
    #[inline]
    fn sigmoid(&self) -> Self {
        1.0f32 / (1.0f32 + (-self).exp())
    }
}
impl Sigmoid for f64 {
    #[inline]
    fn sigmoid(&self) -> Self {
        1. / (1. + (-self).exp())
    }
}

#[macro_export]
macro_rules! expect_cast {
    ($var:expr => $t:path) => {
        if let $t(v) = $var {
            v
        } else {
            unreachable!()
        }
    }
}

pub type LayerT = Vec<AnyNode>;
pub type NetworkLayersT = Vec<LayerT>;
#[derive(Debug)]
pub struct Network {
    shape: Vec<usize>,
    layers: NetworkLayersT,
}
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
                            AnyNode::Normal(
                                Node::new(0., &vec![0.; shape[i - 1]], i))
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
    result_cache: RefCell<Option<f64>>,
}
impl Node {
    pub fn new(bias: f64, inp_w: &Vec<f64>, layer: usize) -> Node {
        let inp_w = inp_w.clone();
        return Self {
            bias,
            inp_w,
            layer,
            result_cache: RefCell::new(None),
        };
    }
    pub fn invalidate(&self) {
        self.result_cache.replace(None);
    }
    
    pub fn get_weight(&self, wi: usize) -> f64 {
        self.inp_w[wi]
    }
    pub fn set_weight(&mut self, wi: usize, v: f64) {
        self.inp_w[wi] = v;
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
        let inp_sum = self.inp_w.iter()
            .enumerate()
            .map(|(i, v)| {
                network.get_node(self.layer - 1, i).get_value(&network) * v})
            .sum::<f64>();
        let val = (inp_sum + self.bias).sigmoid();
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
        assert_eq!(*$network.get_main_node($ni, $li).result_cache.borrow(), $val)
    }
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
    println!("{:#?}", nw);
}

fn main() {
    println!("Hello world!");
    run_checks();
}
