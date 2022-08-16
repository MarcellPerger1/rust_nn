use crate::node::{new_node, AnyNode, Node, StartNode};
use crate::util::{error_deriv, error_f, TryIntoRef, TryIntoRefMut};

pub type LayerT = Vec<AnyNode>;
pub type NetworkLayersT = Vec<LayerT>;
#[derive(Debug)]
pub struct Network {
    pub shape: Vec<usize>,
    pub layers: NetworkLayersT,
}

// main impl
impl Network {
    pub fn new(shape: &Vec<usize>) -> Network {
        let shape = shape.clone();
        assert!(shape.len() >= 2);
        let layers = shape
            .iter()
            .enumerate()
            .map(|(i, n)| {
                (0..*n)
                    .map(|_| -> AnyNode {
                        if i != 0 {
                            new_node(Node::new(0., &vec![0.; shape[i - 1]], i))
                        } else {
                            new_node(StartNode::new(0.))
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
            .map(|(i, n)| error_f(expected[i], n.get_value(&self)))
            .sum()
    }

    pub fn invalidate(&self) {
        let mut li = self.layers.iter();
        li.next();
        for l in li {
            for n in l {
                n.invalidate();
            }
        }
    }
}

// inputs/outputs
impl Network {
    pub fn get_output(&self, i: usize) -> f64 {
        self.layers.last().expect("Network must have layers!")[i].get_value(&self)
    }

    pub fn get_outputs(&self) -> Vec<f64> {
        self.layers
            .last()
            .expect("Network must have layers!")
            .iter()
            .map(|n| n.get_value(&self))
            .collect()
    }

    pub fn set_input(&mut self, i: usize, value: f64) {
        self.get_start_node_mut(i).set_value(value)
    }

    pub fn set_inputs(&mut self, inputs: Vec<f64>) {
        self.layers[0].iter_mut().enumerate().for_each(|(i, n)| {
            n.try_into_ref_mut::<StartNode>()
                .unwrap()
                .set_value(inputs[i])
        });
    }
}

impl Network {
    pub fn request_nudges_end(&self, expected: Vec<f64>) {
        let outputs = self.get_outputs();
        self.layers[self.shape.len() - 1]
            .iter()
            .enumerate()
            .for_each(|(i, n)| {
                let node = n.try_into_ref::<Node>().unwrap();
                let mut want_nudge = error_deriv(outputs[i], expected[i]);
                // want to lower cost = move towards least positive gradient
                want_nudge *= -1.0;
                node.request_nudge(want_nudge);
            });
    }

    pub fn train_on_current_data(&self, expected: Vec<f64>) {
        self.request_nudges_end(expected);
        self.layers.iter().skip(1).rev().for_each(|l| {
            l.iter().for_each(|n| {
                n.try_into_ref::<Node>().unwrap().calc_nudge(self);
            })
        });
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
        &self.layers[i]
    }

    pub fn get_node(&self, li: usize, ni: usize) -> &AnyNode {
        &self.layers[li][ni]
    }
    pub fn get_node_mut(&mut self, li: usize, ni: usize) -> &mut AnyNode {
        &mut self.layers[li][ni]
    }

    pub fn get_start_node(&self, ni: usize) -> &StartNode {
        self.get_node_as(0, ni).unwrap()
    }
    pub fn get_start_node_mut(&mut self, ni: usize) -> &mut StartNode {
        self.get_node_as_mut(0, ni).unwrap()
    }
    pub fn get_main_node(&self, li: usize, ni: usize) -> &Node {
        assert_ne!(li, 0, "no main nodes in layer 0");
        self.get_node_as(li, ni).unwrap()
    }
    pub fn get_main_node_mut(&mut self, li: usize, ni: usize) -> &mut Node {
        assert_ne!(li, 0, "no main nodes in layer 0");
        self.get_node_as_mut(li, ni).unwrap()
    }

    pub fn get_node_as<T: 'static>(&self, li: usize, ni: usize) -> Option<&T> {
        self.get_node(li, ni).try_into_ref()
    }
    pub fn get_node_as_mut<T: 'static>(&mut self, li: usize, ni: usize) -> Option<&mut T> {
        self.get_node_mut(li, ni).try_into_ref_mut()
    }
}
