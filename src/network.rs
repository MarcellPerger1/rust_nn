use crate::util::expect_cast;
use crate::*;

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