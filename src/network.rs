use crate::node::*;
use crate::util::error_f;

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
                    .map(|_| -> AnyNode {
                        if i != 0 {
                            Box::new(Node::new(0., &vec![0.; shape[i - 1]], i)) as Box<dyn NodeLike>
                        } else {
                            Box::new(StartNode::new(0.)) as Box<dyn NodeLike>
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
                    n.get_value(&self),
                )
            })
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
        self.layers[0]
            .iter_mut()
            .enumerate()
            .for_each(|(i, n)| 
                      n.as_any_mut().downcast_mut::<StartNode>().unwrap().set_value(inputs[i]));
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
        self.get_node(0, ni).as_any().downcast_ref().unwrap()
    }
    pub fn get_start_node_mut(&mut self, ni: usize) -> &mut StartNode {
        self.get_node_mut(0, ni).as_any_mut().downcast_mut().unwrap()
    }
    pub fn get_main_node(&self, li: usize, ni: usize) -> &Node {
        assert_ne!(li, 0, "no main nodes in layer 0");
        self.get_node(li, ni).as_any().downcast_ref().unwrap()
    }
    pub fn get_main_node_mut(&mut self, li: usize, ni: usize) -> &mut Node {
        assert_ne!(li, 0, "no main nodes in layer 0");
        self.get_node_mut(li, ni).as_any_mut().downcast_mut().unwrap()
    }
}
