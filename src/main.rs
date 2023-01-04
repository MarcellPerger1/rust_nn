pub mod network;
pub mod node;
pub mod sigmoid;
pub mod training_data;
mod util;

#[cfg(test)]
mod test_util;

#[cfg(test)]
mod main_tests {
    use crate::network::Network;
    use crate::node::NodeLike;
    use crate::test_util::assert_refcell_eq;

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

    #[test]
    fn run_checks() {
        // NOTE TO SELF: remember that activation is passed thru sigmoid activation
        // DONT FORGET THIS when writing tests and wondering why they fail
        // NOTE TO SELF 2: These tests aren't as important as all the unittests
        // as the numbers here aren't actually checked to be correct
        // they are just the result from the previous version
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
        nw.train_on_current_data(&vec![0.5, 0.5]);
        assert_refcell_eq!(nw.get_main_node(1, 1).requested_nudge, -0.4621171572600098);
        assert_refcell_eq!(
            nw.get_main_node(1, 1).inp_w_nudge_sum,
            vec![0.0, -0.09085774767294842]
        );
        assert_refcell_eq!(nw.get_main_node(1, 1).bias_nudge_sum, -0.09085774767294842);
        assert_refcell_eq!(nw.get_main_node(1, 1).nudge_cnt, 1);
        nw.apply_nudges();
        assert_refcell_eq!(nw.get_main_node(1, 1).inp_w_nudge_sum, vec![0.0; 2]);
        assert_refcell_eq!(nw.get_main_node(1, 1).bias_nudge_sum, 0.0);
        assert_refcell_eq!(nw.get_main_node(1, 1).nudge_cnt, 0);
        assert_refcell_eq!(nw.get_main_node(1, 1).requested_nudge, 0.0);
        // <> these values is correct and checked:
        assert_eq!(nw.get_main_node(1, 1).bias, -0.09085774767294842);
        assert_eq!(nw.get_main_node(1, 1).inp_w, vec![0.0, 0.9091422523270516]);
        // </>
        println!("{:#?}", nw);
    }
}

fn main() {
    println!("Hello world!");
}
