use crate::node::{new_node, AnyNode, Node, StartNode, NodeLike};
use crate::training_data::{TrainingData, TrainingExample};
use crate::util::{error_deriv, error_f, TryIntoRef, TryIntoRefMut};

pub type LayerT = Vec<AnyNode>;
pub type NetworkLayersT = Vec<LayerT>;

#[derive(Debug, Clone, PartialEq)]
pub struct NetworkConfig {
    pub learning_rate: f64,
    pub shape: Vec<usize>,
}
impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1.0,
            shape: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct Network {
    pub layers: NetworkLayersT,
    pub config: NetworkConfig,
}

pub trait NodeContainer {
    fn get_node(&self, li: usize, ni: usize) -> &AnyNode;
    fn get_node_mut(&mut self, li: usize, ni: usize) -> &mut AnyNode;
    fn get_node_as<T: 'static>(&self, li: usize, ni: usize) -> Option<&T>{
        self.get_node(li, ni).try_into_ref()
    }
    fn get_node_as_mut<T: 'static>(&mut self, li: usize, ni: usize) -> Option<&mut T> {
        self.get_node_mut(li, ni).try_into_ref_mut()
    }
}
// impl-ing this trait for a type signals that in that type,
// start nodes are in first layer and Nodes are in main layer
pub trait LayeredNetwork: NodeContainer {
    fn get_start_node(&self, ni: usize) -> &StartNode {
        self.get_node_as(0, ni).unwrap()
    }
    fn get_start_node_mut(&mut self, ni: usize) -> &mut StartNode {
        self.get_node_as_mut(0, ni).unwrap()
    }
    fn get_main_node(&self, li: usize, ni: usize) -> &Node {
        assert_ne!(li, 0, "no main nodes in layer 0");
        self.get_node_as(li, ni).unwrap()
    }
    fn get_main_node_mut(&mut self, li: usize, ni: usize) -> &mut Node {
        assert_ne!(li, 0, "no main nodes in layer 0");
        self.get_node_as_mut(li, ni).unwrap()
    }
}

pub trait InOutNetwork {
    fn get_output(&self, i: usize) -> f64;
    fn get_outputs(&self) -> Vec<f64>;

    fn set_input(&mut self, i: usize, value: f64);
    fn set_inputs(&mut self, inputs: &Vec<f64>);
}

// the implementor can almost ceratinly provide better implementations of some of these methods
pub trait InOutLayeredNetwork: InOutNetwork + LayeredNetwork {
    fn layers_ref(&self) -> &NetworkLayersT;
    fn layers_mut(&mut self) -> &mut NetworkLayersT;
    
    fn shape_vec(&self) -> Vec<usize> {
        self.layers_ref().iter().map(|l| l.len()).collect()
    }
    fn n_layers(&self) -> usize {
        self.layers_ref().len()
    }
    fn last_layer(&self) -> &LayerT {
        self.layers_ref().last().expect("Network must have layers")
    }

    fn get_output(&self, i: usize) -> f64 {
        0.0
        // self.get_main_node(self.n_layers() - 1, i).get_value(&self)
    }

    fn get_outputs(&self) -> Vec<f64> {
        vec![]
        // self.last_layer()
        //     .iter()
        //     .map(|n| n.get_value(&self))
        //     .collect()
    }

    fn set_input(&mut self, i: usize, value: f64) {
        self.get_start_node_mut(i).set_value(value)
    }

    fn set_inputs(&mut self, inputs: &Vec<f64>) {
        assert_eq!(
            inputs.len(), 
            self.shape_vec()[0], 
            "number of inputs must match number of nodes in layer 0");
        self.layers_mut()[0].iter_mut().enumerate().for_each(|(i, n)| {
            n.try_into_ref_mut::<StartNode>()
                .unwrap()
                .set_value(inputs[i])
        });
    }
}


// main impl
impl Network {
    pub fn new(shape: &Vec<usize>) -> Network {
        let shape = shape.clone();
        return Self::with_config(&NetworkConfig {
            shape,
            ..Default::default()
        });
    }

    pub fn with_config(config: &NetworkConfig) -> Network {
        let config = config.clone();
        assert!(
            config.shape.len() >= 2,
            "network must have at least start and end layers!"
        );
        let layers = config
            .shape
            .iter()
            .enumerate()
            .map(|(i, n)| {
                (0..*n)
                    .map(|_| -> AnyNode {
                        if i != 0 {
                            new_node(Node::new(0., &vec![0.; config.shape[i - 1]], i))
                        } else {
                            new_node(StartNode::new(0.))
                        }
                    })
                    .collect::<LayerT>()
            })
            .collect();
        Network { config, layers }
    }

    pub fn get_current_cost(&self, expected: &Vec<f64>) -> f64 {
        assert_eq!(
            self.last_layer().len(),
            expected.len(),
            "Length of expected must match length of output"
        );
        self.layers[self.layers.len() - 1]
            .iter()
            .enumerate()
            .map(|(i, n)| error_f(expected[i], n.get_value(&self)))
            .sum()
    }

    pub fn invalidate(&self) {
        self.layers.iter().skip(1).for_each(|l| {
            l.iter().for_each(|n| {
                n.invalidate();
            })
        });
    }
}

// inputs/outputs
impl Network {
    pub fn get_output(&self, i: usize) -> f64 {
        self.last_layer()[i].get_value(&self)
    }

    pub fn get_outputs(&self) -> Vec<f64> {
        self.last_layer()
            .iter()
            .map(|n| n.get_value(&self))
            .collect()
    }

    pub fn set_input(&mut self, i: usize, value: f64) {
        self.get_node_as_mut::<StartNode>(0, i).unwrap().set_value(value)
    }

    pub fn set_inputs(&mut self, inputs: &Vec<f64>) {
        assert_eq!(
            inputs.len(), 
            self.layers[0].len(), 
            "number of inputs must match number of nodes in layer 0");
        self.layers[0].iter_mut().enumerate().for_each(|(i, n)| {
            n.try_into_ref_mut::<StartNode>()
                .unwrap()
                .set_value(inputs[i])
        });
    }
}

impl Network {
    pub fn request_nudges_end(&self, expected: &Vec<f64>) {
        let outputs = self.get_outputs();
        self.layers[self.layers.len() - 1]
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

    pub fn train_on_current_data(&self, expected: &Vec<f64>) {
        self.request_nudges_end(expected);
        self.layers.iter().skip(1).rev().for_each(|l| {
            l.iter().for_each(|n| {
                n.try_into_ref::<Node>().unwrap().calc_nudge(self);
            })
        });
    }

    pub fn train_on_data(&mut self, data: &TrainingExample) {
        self.set_inputs(&data.inputs);
        self.invalidate();
        self.train_on_current_data(&data.expected);
    }

    pub fn clear_nudges(&mut self) {
        self.layers.iter_mut().skip(1).for_each(|l| {
            l.iter_mut().for_each(|n| {
                n.try_into_ref_mut::<Node>().unwrap().clear_nudges();
            })
        })
    }

    pub fn apply_nudges(&mut self) {
        self.layers.iter_mut().skip(1).for_each(|l| {
            l.iter_mut().for_each(|n| {
                let n = n.try_into_ref_mut::<Node>().unwrap();
                n.apply_nudges();
                n.clear_nudges();
            })
        });
    }

    pub fn train_on_batch(&mut self, batch: &TrainingData) {
        batch.0.iter().for_each(|data| self.train_on_data(data));
        self.apply_nudges();
    }

    pub fn train_on_batches(&mut self, batches: &Vec<TrainingData>) {
        batches.iter().for_each(|b| {
            self.train_on_batch(b);
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

    #[inline]
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    #[inline]
    pub fn last_layer(&self) -> &LayerT {
        self.layers.last().expect("Network must have layers!!!")
    }
}
use mockall::mock;
mock! {
    pub Network {
        pub fn new(shape: &Vec<usize>) -> MockNetwork;
        pub fn with_config(config: &NetworkConfig) -> MockNetwork;
        pub fn get_current_cost(&self, expected: &Vec<f64>) -> f64;
        pub fn invalidate(&self);
        pub fn get_output(&self, i: usize) -> f64;
        pub fn get_outputs(&self) -> Vec<f64>;
        pub fn set_input(&mut self, i: usize, value: f64);
        pub fn set_inputs(&mut self, inputs: &Vec<f64>);
        pub fn request_nudges_end(&self, expected: &Vec<f64>);
        pub fn train_on_current_data(&self, expected: &Vec<f64>);
        pub fn train_on_data(&mut self, data: &TrainingExample);
        pub fn clear_nudges(&mut self);
        pub fn apply_nudges(&mut self);
        pub fn train_on_batch(&mut self, batch: &TrainingData);
        pub fn train_on_batches(&mut self, batches: &Vec<TrainingData>);
        pub fn layers_ref(&self) -> &NetworkLayersT;
        pub fn layers_mut(&mut self) -> &mut NetworkLayersT;
        pub fn get_layer(&self, i: usize) -> &LayerT;
        pub fn get_node(&self, li: usize, ni: usize) -> &AnyNode;
        pub fn get_node_mut(&mut self, li: usize, ni: usize) -> &mut AnyNode;
        pub fn get_start_node(&self, ni: usize) -> &StartNode;
        pub fn get_start_node_mut(&mut self, ni: usize) -> &mut StartNode;
        pub fn get_main_node(&self, li: usize, ni: usize) -> &Node;
        pub fn get_main_node_mut(&mut self, li: usize, ni: usize) -> &mut Node;
        pub fn get_node_as<T: 'static>(&self, li: usize, ni: usize) -> Option<&'static T>;
        pub fn get_node_as_mut<T: 'static>(&mut self, li: usize, ni: usize) -> Option<&'static mut T>;
        pub fn n_layers(&self) -> usize;
        pub fn last_layer(&self) -> &LayerT;
    }
}
#[cfg(test)]
mod tests;
