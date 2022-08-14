pub mod network;
pub mod node;
pub mod sigmoid;
mod util;

use crate::network::Network;
use crate::node::NodeValue;

macro_rules! assert_cached_eq {
    ($network:expr, $ni:expr, $li:expr, $val:expr) => {
        assert_eq!(
            *$network.get_main_node($ni, $li).result_cache.borrow(),
            $val
        )
    };
    ($cache:expr, $val:expr) => {
        assert_eq!(*$cache.borrow(), $val)
    };
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
    assert_cached_eq!(nw.get_main_node(1, 1).sum_cache, Some(1.));
    assert_eq!(nw.get_current_cost(&vec![0.5, 1.0]), 0.07232948812851325);
    println!("{:#?}", nw);
}

fn main() {
    println!("Hello world!");
    run_checks();
}
