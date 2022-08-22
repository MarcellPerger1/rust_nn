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
// TODO use mockall crate for mocking???
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
