use crate::network::Network;
use crate::sigmoid::Sigmoid;
use crate::util::{impl_as_any, AsAny, TryIntoRef, TryIntoRefMut};
use std::cell::RefCell;

pub type AnyNode = Box<dyn NodeLike>;
pub type AnyNodeLT<'a> = Box<dyn NodeLike + 'a>;

pub trait NodeLike: AsAny + std::fmt::Debug {
    fn get_value(&self, network: &Network) -> f64;

    fn invalidate(&self) {
        // do nothing by default
    }

    fn request_nudge(&self, _nudge: f64) {}
}

impl TryIntoRef for dyn NodeLike {
    fn try_into_ref<T: 'static>(&self) -> Option<&T> {
        self.as_any().downcast_ref()
    }
}
impl TryIntoRefMut for dyn NodeLike {
    fn try_into_ref_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.as_any_mut().downcast_mut()
    }
}

pub fn new_node<'a, T>(n: T) -> AnyNodeLT<'a>
where
    T: NodeLike + 'a,
{
    Box::new(n) as AnyNodeLT
}

#[derive(Debug)]
pub struct Node {
    pub bias: f64,
    pub inp_w: Vec<f64>,
    pub layer: usize,
    pub(crate) sum_cache: RefCell<Option<f64>>,
    pub(crate) result_cache: RefCell<Option<f64>>,
    // backpropagation
    pub bias_nudge_sum: RefCell<f64>,
    pub inp_w_nudge_sum: RefCell<Vec<f64>>,
    pub nudge_cnt: RefCell<usize>,
    pub requested_nudge: RefCell<f64>,
}
impl_as_any!(Node);
impl Node {
    pub fn new(bias: f64, inp_w: &Vec<f64>, layer: usize) -> Node {
        return Self {
            bias,
            inp_w: inp_w.clone(),
            layer,
            result_cache: RefCell::new(None),
            sum_cache: RefCell::new(None),
            // backpropagation
            bias_nudge_sum: RefCell::new(0.0),
            inp_w_nudge_sum: RefCell::new(vec![0.0; inp_w.len()]),
            nudge_cnt: RefCell::new(0),
            requested_nudge: RefCell::new(0.0),
        };
    }

    pub fn is_last_layer(&self, network: &Network) -> bool {
        self.layer == network.n_layers() - 1
    }

    pub fn get_weight(&self, wi: usize) -> f64 {
        self.inp_w[wi]
    }
    pub fn set_weight(&mut self, wi: usize, v: f64) {
        self.inp_w[wi] = v;
    }

    pub fn calc_nudge(&self, network: &Network) {
        let d_sig = self.get_sum(network).sig_deriv();
        let base_nudge = *self.requested_nudge.borrow() * d_sig * network.config.learning_rate;
        // bias nudge
        *self.bias_nudge_sum.borrow_mut() += base_nudge;
        (0..self.inp_w.len()).for_each(|i| {
            // weight nudges
            (*self.inp_w_nudge_sum.borrow_mut())[i] +=
                base_nudge * network.get_node(self.layer - 1, i).get_value(network);
            
            // nudge previous nodes
            // this doesn't appear to work: WHY?
            // network
            //     .get_node(self.layer - 1, i)
            //     .request_nudge(base_nudge * self.inp_w[i]);
            if self.layer > 1 {
                *network.get_node(self.layer - 1, i)
                    .try_into_ref::<Node>().unwrap()
                    .requested_nudge.borrow_mut() += base_nudge * self.inp_w[i];
            }
        });
        (*self.nudge_cnt.borrow_mut()) += 1;
    }

    pub fn request_nudge(&self, nudge: f64) {
        *self.requested_nudge.borrow_mut() += nudge;
    }

    pub fn apply_nudges(&mut self) {
        let inv_cnt = 1.0 / (*self.nudge_cnt.borrow() as f64);
        self.bias += *self.bias_nudge_sum.borrow() * inv_cnt;
        self.inp_w_nudge_sum
            .borrow()
            .iter()
            .enumerate()
            .for_each(|(i, wn)| {
                self.inp_w[i] += wn * inv_cnt;
            });
    }

    pub fn clear_nudges(&self) {
        self.nudge_cnt.replace(0);
        self.bias_nudge_sum.replace(0.0);
        // or .borrow_mut().fill(0.0)
        self.inp_w_nudge_sum.replace(vec![0.0; self.inp_w.len()]);
        self.requested_nudge.replace(0.0);
    }

    pub fn get_sum(&self, network: &Network) -> f64 {
        if let Some(cached) = *self.sum_cache.borrow() {
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
impl NodeLike for Node {
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

    fn invalidate(&self) {
        self.sum_cache.replace(None);
        self.result_cache.replace(None);
    }
}

#[derive(Debug)]
pub struct StartNode {
    pub(crate) value: f64,
}
impl_as_any!(StartNode);
impl NodeLike for StartNode {
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

    pub fn get_start_value(&self) -> f64 {
        // same as `get_value` but doesn't require `Network` as argument
        // the different naming is because rust can't differentiate properly
        // between a 0-arg `get_value` defined on the struct and a 1-arg
        // `get_value` defined on the `NodeLike` trait
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::assert_refcell_eq;
    use crate::network::NetworkConfig;

    mod start_node {
        use super::*;
        use crate::network::Network;

        #[test]
        fn test_new() {
            let n = StartNode::new(0.7);
            assert_eq!(n.value, 0.7);
        }

        #[test]
        fn set_value() {
            let mut n = StartNode::new(3.3);
            n.set_value(0.1);
            assert_eq!(n.value, 0.1);
        }

        #[test]
        fn get_value() {
            // not very clean, but `get_value` needs a argument of type `Network`
            let mut nw = Network::new(&vec![1, 1]);
            nw.get_start_node_mut(0).value = -1.9;
            assert_eq!(nw.get_node(0, 0).get_value(&nw), -1.9);
        }

        #[test]
        fn get_value_start() {
            let n = StartNode::new(8.7);
            assert_eq!(n.get_start_value(), 8.7);
        }
    }

    mod main_node {
        use super::*;

        /// Makes network with shape `[3, 5, 4, 2]`
        fn make_nw() -> Network {
            return Network::new(&vec![3, 5, 4, 2]);
        }

        #[test]
        fn new_params() {
            let n = Node::new(2.1, &vec![0.4, -7.4], 4);
            assert_eq!(n.bias, 2.1);
            assert_eq!(n.inp_w, vec![0.4, -7.4]);
            assert_eq!(n.layer, 4);
        }

        #[test]
        fn new_cache() {
            let n = Node::new(2.1, &vec![0.4, -7.4], 4);
            assert_refcell_eq!(n.result_cache, None);
            assert_refcell_eq!(n.result_cache, None);
        }

        #[test]
        fn new_nudges() {
            let n = Node::new(2.1, &vec![0.4, -7.4], 4);
            assert_refcell_eq!(n.nudge_cnt, 0);
            assert_refcell_eq!(n.bias_nudge_sum, 0.0);
            assert_refcell_eq!(n.inp_w_nudge_sum, vec![0.0; 2]);
        }

        #[test]
        fn is_last_layer_in_network() {
            let nw = Network::new(&vec![1, 2, 3]);
            let n = nw.get_main_node(2, 1);
            assert_eq!(n.is_last_layer(&nw), true);
            let n = nw.get_main_node(1, 0);
            assert_eq!(n.is_last_layer(&nw), false);
        }

        #[test]
        fn is_last_layer_indep() {
            let nw = Network::new(&vec![1, 2, 3]);
            let n = Node::new(0.2, &vec![2.1, -0.3], 2);
            assert_eq!(n.is_last_layer(&nw), true);
            let n = Node::new(0.2, &vec![-9.8], 1);
            assert_eq!(n.is_last_layer(&nw), false);
        }

        mod inp_w {
            use super::*;

            #[test]
            fn get_weight() {
                let n = Node::new(0.0, &vec![1.0, -0.2, 0.0], 1);
                assert_eq!(n.get_weight(0), 1.0);
                assert_eq!(n.get_weight(2), 0.0);
            }

            #[test]
            #[should_panic]
            fn get_weight_oob() {
                let n = Node::new(0.0, &vec![1.0, -0.2, 0.0], 1);
                n.get_weight(3);
            }

            #[test]
            fn set_weight() {
                let mut n = Node::new(0.0, &vec![1.0, -0.2, 0.0], 1);
                n.set_weight(2, 3.4);
                assert_eq!(n.inp_w[2], 3.4);
            }

            #[test]
            #[should_panic]
            fn set_weight_oob() {
                let mut n = Node::new(0.0, &vec![1.0, -0.2, 0.0], 1);
                n.set_weight(3, 3.4);
            }
        }

        #[test]
        fn invalidate_sum_cache() {
            let n = Node::new(2.1, &vec![0.4, -7.4], 2);
            n.sum_cache.replace(Some(-0.8));
            n.invalidate();
            assert_refcell_eq!(n.sum_cache, None);
        }

        #[test]
        fn invalidate_result_cache() {
            let n = Node::new(2.1, &vec![0.4, -7.4], 2);
            n.result_cache.replace(Some(-0.8));
            n.invalidate();
            assert_refcell_eq!(n.result_cache, None);
        }

        mod get_sum {
            use super::*;

            #[test]
            fn uses_cached() {
                let nw = make_nw();
                let n = nw.get_main_node(2, 1);
                n.sum_cache.replace(Some(-77.6));
                assert_eq!(n.get_sum(&nw), -77.6);
            }

            #[test]
            fn sums_prev_layer_and_bias() {
                let mut nw = make_nw();
                nw.set_inputs(&vec![0.4, 0.8, 0.1]);
                let mut n = nw.get_main_node_mut(1, 2);
                n.bias = 0.5;
                n.inp_w = vec![2.3, -1.2, 0.6];
                n.sum_cache.replace(None); // ensure not cached
                let n = nw.get_main_node(1, 2);
                let expected = 0.4 * 2.3 - 0.8 * 1.2 + 0.1 * 0.6 + 0.5;
                assert_eq!(n.get_sum(&nw), expected);
            }

            #[test]
            fn sets_cache() {
                let mut nw = make_nw();
                nw.set_inputs(&vec![0.4, 0.8, 0.1]);
                let mut n = nw.get_main_node_mut(1, 2);
                n.bias = 0.5;
                n.inp_w = vec![2.3, -1.2, 0.6];
                n.sum_cache.replace(None); // ensure not cached
                let n = nw.get_main_node(1, 2);
                let value = n.get_sum(&nw);
                assert_refcell_eq!(n.sum_cache, Some(value));
            }
        }

        mod get_value {
            use super::*;

            #[test]
            fn uses_cached() {
                let nw = make_nw();
                let n = nw.get_main_node(2, 1);
                n.result_cache.replace(Some(0.9031));
                assert_eq!(n.get_value(&nw), 0.9031);
            }

            #[test]
            fn uses_cached_sum() {
                let nw = make_nw();
                let n = nw.get_main_node(2, 1);
                n.sum_cache.replace(Some(5.67));
                assert_eq!(n.get_value(&nw), (5.67).sigmoid());
            }

            #[test]
            fn sets_cache() {
                let nw = make_nw();
                let n = nw.get_main_node(2, 1);
                n.sum_cache.replace(Some(5.67));
                let value = n.get_value(&nw);
                assert_refcell_eq!(n.result_cache, Some(value));
            }

            #[test]
            fn works_without_cache() {
                let mut nw = make_nw();
                nw.set_inputs(&vec![0.9, 0.0, 0.1]);
                let mut n = nw.get_main_node_mut(1, 2);
                n.bias = -3.5;
                n.inp_w = vec![2.3, -1.2, 0.6];
                n.invalidate(); // ensure nothing cached
                let n = nw.get_main_node(1, 2);
                let expected = (0.9 * 2.3 - 0.0 * 1.2 + 0.1 * 0.6 - 3.5).sigmoid();
                assert_eq!(n.get_value(&nw), expected);
            }
        }

        #[test]
        fn request_nudge() {
            let n = Node::new(0.2, &vec![2.1, -0.3], 1);
            n.request_nudge(3.1);
            assert_refcell_eq!(n.requested_nudge, 3.1);
            n.request_nudge(-0.8);
            assert_refcell_eq!(n.requested_nudge, 3.1 - 0.8);
        }

        #[test]
        fn clear_nudges() {
            let n = Node::new(1.2, &vec![2.2, -0.33], 2);
            n.bias_nudge_sum.replace(7.998);
            n.inp_w_nudge_sum.replace(vec![3.1, -2.8]);
            n.requested_nudge.replace(-3.02);
            n.nudge_cnt.replace(6);
            n.clear_nudges();
            assert_refcell_eq!(n.nudge_cnt, 0);
            assert_refcell_eq!(n.bias_nudge_sum, 0.0);
            assert_refcell_eq!(n.inp_w_nudge_sum, vec![0.0; 2]);
            assert_refcell_eq!(n.requested_nudge, 0.0);
        }

        mod calc_nudge {
            use super::*;

            fn layer1_template(lr: f64) {
                let mut nw = Network::with_config(&NetworkConfig {
                    learning_rate: lr,
                    shape: vec![3, 2]
                });
                nw.set_inputs(&vec![0.8, 0.12, 0.53]);
                let n = nw.get_main_node(1, 1);
                n.request_nudge(0.9);
                n.sum_cache.replace(Some(-0.8));
                n.calc_nudge(&nw);
                let base_nudge = 0.9 * (-0.8).sig_deriv() * lr;
                assert_refcell_eq!(n.bias_nudge_sum, base_nudge);
                assert_refcell_eq!(n.inp_w_nudge_sum, vec![
                    base_nudge * 0.8, base_nudge * 0.12, base_nudge * 0.53]);
                assert_refcell_eq!(n.nudge_cnt, 1);
            }

            #[test]
            fn no_prev_layer() {
                layer1_template(1.0);
            }
            #[test]
            fn no_prev_learning_rate() {
                layer1_template(2.7);
                layer1_template(0.3);
            }
    
            fn layer2_template(lr: f64) {
                let mut nw = Network::with_config(&NetworkConfig {
                    learning_rate: lr,
                    shape: vec![5, 5, 3]
                });
                let layer_1_value = vec![0.2, 0.9, 0.0, 0.3, 1.0];
                for (i, n) in nw.layers[1].iter().enumerate() {
                    n.try_into_ref::<Node>()
                        .unwrap()
                        .result_cache.replace(Some(layer_1_value[i]));
                }
                let n = nw.get_main_node_mut(2, 1);
                let inp_w = vec![3.5, 0.3, -2.1, 0.0, -4.7];
                n.inp_w.clone_from(&inp_w);
                let n = nw.get_main_node(2, 1);
                n.request_nudge(-0.3);
                n.sum_cache.replace(Some(1.8));
                n.calc_nudge(&nw);
                let base_nudge = -0.3 * (1.8).sig_deriv() * lr;
                assert_refcell_eq!(n.bias_nudge_sum, base_nudge);
                assert_refcell_eq!(n.inp_w_nudge_sum, layer_1_value.iter().map(|nv| {
                    base_nudge * nv
                }).collect::<Vec<_>>());
                let actual: Vec<_> = nw.layers[1].iter().map(|n| {
                    *n.try_into_ref::<Node>().unwrap().requested_nudge.borrow()
                }).collect();
                assert_eq!(actual, inp_w.iter().map(|w| {
                    base_nudge * w
                }).collect::<Vec<_>>());
                assert_refcell_eq!(n.nudge_cnt, 1);
            }

            #[test]
            fn has_prev_layer() {
                layer2_template(1.0);
            }
            #[test]
            fn has_prev_learning_rate() {
                layer2_template(3.1);
                layer2_template(0.47);
            }
        }
    }
}
