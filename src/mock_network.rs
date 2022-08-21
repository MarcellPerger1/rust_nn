use mockall::mock;

use crate::network::{NetworkConfig, NetworkLayersT, LayerT};
//pub use crate::network::Network;
use crate::training_data::*;
use crate::node::*;
use crate::mock_network::MockNetwork as Network;

mock! {
    pub Network {
        pub fn new(shape: &Vec<usize>) -> Network;
        pub fn with_config(config: &NetworkConfig) -> Network;
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

