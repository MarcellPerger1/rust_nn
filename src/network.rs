use crate::node::{new_node, AnyNode, Node, StartNode};
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
        self.get_start_node_mut(i).set_value(value)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::*;

    #[test]
    fn init_by_shape() {
        let shape = vec![5, 3, 2];
        let nw = Network::new(&shape);
        assert_eq!(nw.layers.len(), shape.len());
        nw.layers.iter().enumerate().for_each(|(i, l)| {
            assert_eq!(l.len(), shape[i]);
        })
    }
    #[test]
    #[should_panic(expected = "network must have at least start and end layers")]
    fn few_layers_fast_fail() {
        Network::new(&vec![7]);
    }
    #[test]
    fn nodes_know_self_layer() {
        let shape = vec![5, 3, 2];
        let nw = Network::new(&shape);
        nw.layers.iter().enumerate().skip(1).for_each(|(i, l)| {
            l.iter().for_each(|n| {
                assert_eq!(n.try_into_ref::<Node>().unwrap().layer, i);
            })
        })
    }
    #[test]
    fn nodes_have_correct_inp_w_length() {
        let shape = vec![5, 3, 2];
        let nw = Network::new(&shape);
        nw.layers.iter().enumerate().skip(1).for_each(|(i, l)| {
            l.iter().for_each(|n| {
                assert_eq!(n.try_into_ref::<Node>().unwrap().inp_w.len(), shape[i - 1]);
            })
        })
    }
    #[test]
    fn start_layer_has_start_nodes() {
        let shape = vec![5, 3, 2];
        let nw = Network::new(&shape);
        nw.layers[0].iter().for_each(|n| {
            n.try_into_ref::<StartNode>()
                .expect("Start layer must only conatin start nodes");
        })
    }
    #[test]
    fn main_layers_have_main_nodes() {
        let shape = vec![5, 3, 2];
        let nw = Network::new(&shape);
        nw.layers.iter().skip(1).for_each(|l| {
            l.iter().for_each(|n| {
                n.try_into_ref::<Node>()
                    .expect("Main layer must only conatin main nodes");
            })
        })
    }
    #[test]
    fn config_built_correctly() {
        let shape = vec![5, 3, 2];
        let nw = Network::new(&shape);
        assert_eq!(
            nw.config,
            NetworkConfig {
                shape: vec![5, 3, 2],
                ..Default::default()
            }
        );
    }
    #[test]
    fn config_arg_respected() {
        let config = NetworkConfig {
            learning_rate: 3.5,
            shape: vec![10, 6, 3],
        };
        let nw = Network::with_config(&config);
        assert_eq!(nw.config, config);
    }
    fn new_nw() -> Network {
        Network::new(&vec![5, 3, 2])
    }
    #[test]
    #[should_panic(expected = "Length of expected must match length of output")]
    fn current_cost_bad_expected_fails() {
        new_nw().get_current_cost(&vec![0.1; 5]);
    }
    #[test]
    fn current_cost_zero() {
        let mut nw = new_nw();
        nw.set_inputs(&vec![0.0; 5]);
        assert_eq!(nw.get_current_cost(&vec![0.5; 2]), 0.0);
    }
    #[test]
    fn current_cost_normal() {
        let mut nw = Network::new(&vec![7, 5, 5, 2]);
        nw.set_inputs(&vec![0.0; 7]);
        assert_f_eq!(nw.get_current_cost(&vec![0.9, 0.31]), 0.1961);
    }
    #[test]
    fn test_invaildate() {
        let nw = new_nw();
        // set caches
        nw.get_outputs();
        nw.layers.iter().skip(1).for_each(|l| {
            l.iter().for_each(|n| {
                let n: &Node = n.try_into_ref().unwrap();
                assert!(n.sum_cache.borrow().is_some());
                assert!(n.result_cache.borrow().is_some());
            })
        });
        nw.invalidate();
        nw.layers.iter().skip(1).for_each(|l| {
            l.iter().for_each(|n| {
                let n: &Node = n.try_into_ref().unwrap();
                assert!(n.sum_cache.borrow().is_none());
                assert!(n.result_cache.borrow().is_none());
            })
        });
    }
    #[test]
    fn test_set_input() {
        let mut nw = new_nw();
        nw.set_input(3, 0.7);
        assert_eq!(nw.layers[0][3].get_value(&nw), 0.7);
    }
    #[test]
    #[should_panic(expected = "6")]
    fn set_input_oob() { // (out of bounds)
        let mut nw = new_nw();
        nw.set_input(6, 0.7);
    }
    #[test]
    fn test_set_inputs() {
        let mut nw = new_nw();
        let inps = vec![0.2; 5];
        nw.set_inputs(&inps);
        nw.layers[0].iter().enumerate().for_each(|(i, n)| {
            assert_eq!(n.get_value(&nw), inps[i]);
        })
    }
    #[test]
    #[should_panic(expected = "number of inputs must match number of nodes in layer 0")]
    fn set_inputs_too_long() {
        let mut nw = new_nw();
        let inps = vec![0.8; 6];
        nw.set_inputs(&inps);
    }
    #[test]
    #[should_panic(expected = "number of inputs must match number of nodes in layer 0")]
    fn set_inupts_too_short() {
        let mut nw = new_nw();
        let inps = vec![0.9; 4];
        nw.set_inputs(&inps);
    }
    #[test]
    fn get_output_from_cache_forced() {
        let nw = new_nw();
        (0..*nw.config.shape.last().unwrap()).for_each(|i| {
            nw.last_layer()[i]
                .try_into_ref::<Node>()
                .unwrap()
                .result_cache
                .replace(Some(-4.5));
            assert_eq!(nw.get_output(i), -4.5);
        });
    }
    #[test]
    fn get_outputs_from_cache_forced() {
        let nw = new_nw();
        let out_cnt = *nw.config.shape.last().unwrap();
        (0..out_cnt).for_each(|i| {
            nw.last_layer()[i]
                .try_into_ref::<Node>()
                .unwrap()
                .result_cache
                .replace(Some(0.2));
        });
        assert_eq!(nw.get_outputs(), vec![0.2; out_cnt]);
    }
}
