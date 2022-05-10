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

#[derive(Debug)]
pub struct Network {
    shape: Vec<usize>,
    layers: Vec<Vec<AnyNode>>,
}
impl Network {
    pub fn new(shape: &Vec<usize>) -> Network {
        // im afraid its a chicken and egg situation: 
        // Node needs Network to exists (to get a reference)
        // but Network needs those Nodes in the initialisation
        // solution: set layers to None first then once Nodes created,
        // set to actual value
        let shape = shape.clone();
        let layers = shape
            .iter()
            .enumerate()
            .map(|(i, n)| {
                (0..*n)
                    .map(|_| {
                        if i != 0 {
                            AnyNode::Normal(Node::new(0., &Vec::from([])))
                        } else {
                            AnyNode::Start(StartNode::new(0.))
                        }
                    })
                    .collect::<Vec<AnyNode>>()
            })
            .collect();
        let nw = Network { shape, layers };
        return nw;
        
    }
    
    pub fn layers_ref(& self) -> &Vec<Vec<AnyNode>>{
        &self.layers
    }
    pub fn get_layer(&self, i: usize) -> &Vec<AnyNode>{
        &self.layers_ref()[i]
    }
    pub fn get_node(&self, li: usize, ni: usize) -> &AnyNode{
        &self.get_layer(li)[ni]
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
    result_cache: RefCell<Option<f64>>,
}
impl Node {
    pub fn new(bias: f64, inp_w: &Vec<f64>) -> Node {
        let inp_w = inp_w.clone();
        return Self {
            bias,
            inp_w,
            result_cache: RefCell::new(None),
        };
    }
    fn invalidate(&self) {
        self.result_cache.replace(None);
    }
}

impl NodeValue for Node {
    fn get_value(&self, network: &Network) -> f64 {
        if let Some(cached) = *self.result_cache.borrow() {
            return cached;
        }
        let val = ((&self.inp_w)
            .iter()
            .enumerate()
            .map(|(i, v)| network.get_node(0, i).get_value(&network) * v)
            .sum::<f64>()
            + self.bias)
            .sigmoid();
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
}
//endregion

fn run_checks() {
    let nw = Network::new(&vec![2,2]);
    let n = Node::new(0.2, &vec![1.0, 2.0]);
    //println!("{:#?}", n);
    let expect_v = 0.549833997312478;
    let v = n.get_value(&nw);
    //println!("{:#?}", n);
    assert_eq!(v, expect_v);
    assert_eq!(*n.result_cache.borrow(), Some(expect_v));
    let v = n.get_value(&nw);
    assert_eq!(v, expect_v);
    n.invalidate();
    assert_eq!(*n.result_cache.borrow(), None);
}

fn main() {
    println!("Hello world!");
    run_checks();
}
