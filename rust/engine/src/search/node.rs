use crate::{game::board::Loc, inference::outputs::NNOutput};
use std::{ptr::NonNull, sync::Arc};
pub(crate) struct SearchNode {
    nn_output: Option<Arc<NNOutput>>,
    visits: i32,
    white_win_sum: f64, //higher precision
    white_score_sum: f64,
    white_score_mean_sq_sum: f64,
    children: ChildStorage,
}

struct ChildStorage(Vec<Edge>); //naive first version, postpone staged storage opt

struct Edge {
    move_loc: Loc,
    visits: i32,       // used for graph search
    policy_prior: f32, //initial policy prob
    child: NonNull<SearchNode>,
}

impl SearchNode {
    pub(crate) fn new() -> Self {
        Self {
            nn_output: None,
            visits: 0,
            white_win_sum: 0.0,
            white_score_sum: 0.0,
            white_score_mean_sq_sum: 0.0,
            children: ChildStorage::new(),
        }
    }
    pub(crate) fn attach_nn_output(&mut self, output: Arc<NNOutput>) {
        debug_assert!(self.nn_output.is_none(), "only populate empty nodes");
        debug_assert!(output.is_processed());
        self.nn_output = Some(output);
    }
    pub(crate) fn record_visit(
        &mut self,
        white_win: f64,
        white_score: f64,
        white_score_mean_sq: f64,
    ) {
        self.visits += 1;
        self.white_win_sum += white_win;
        self.white_score_sum += white_score;
        self.white_score_mean_sq_sum += white_score_mean_sq;
    }
    pub(crate) fn add_child(
        &mut self,
        move_loc: Loc,
        policy_prior: f32,
        child: NonNull<SearchNode>,
    ) -> usize {
        debug_assert!(policy_prior.is_finite());
        debug_assert!((0.0..=1.0).contains(&policy_prior));

        let edge_index = self.children.0.len();

        self.children.0.push(Edge {
            move_loc,
            visits: 0, //visits get incremented during backup
            policy_prior,
            child,
        });

        edge_index
    }
}
impl ChildStorage {
    fn new() -> Self {
        Self(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{game::board::Player, inference::policy::POLICY_SIZE};

    fn processed_output() -> Arc<NNOutput> {
        let mut output = Arc::new(NNOutput::from_raw([0.0; POLICY_SIZE], 0.0, 0.0, 0.0));
        Arc::get_mut(&mut output)
            .expect("test output is exclusively owned")
            .process_in_place(Player::White, &[true; POLICY_SIZE]);
        output
    }

    #[test]
    fn new_search_node_is_unevaluated_and_unvisited() {
        let node = SearchNode::new();

        assert!(node.nn_output.is_none());
        assert_eq!(node.visits, 0);
        assert_eq!(node.white_win_sum, 0.0);
        assert_eq!(node.white_score_sum, 0.0);
        assert_eq!(node.white_score_mean_sq_sum, 0.0);
        assert!(node.children.0.is_empty());
    }

    #[test]
    fn output_attachment_and_visit_recording_update_separate_state() {
        let mut node = SearchNode::new();
        let output = processed_output();

        node.attach_nn_output(output.clone());
        node.record_visit(0.75, 3.5, 14.0);
        node.record_visit(0.25, -1.5, 6.0);

        assert!(Arc::ptr_eq(node.nn_output.as_ref().unwrap(), &output));
        assert_eq!(node.visits, 2);
        assert_eq!(node.white_win_sum, 1.0);
        assert_eq!(node.white_score_sum, 2.0);
        assert_eq!(node.white_score_mean_sq_sum, 20.0);
    }

    #[test]
    fn adding_children_returns_stable_edge_indices_with_zero_visits() {
        let mut parent = SearchNode::new();
        let mut first_child = Box::new(SearchNode::new());
        let mut second_child = Box::new(SearchNode::new());
        let first_ptr = NonNull::from(first_child.as_mut());
        let second_ptr = NonNull::from(second_child.as_mut());
        let first_move = Loc::new(3, 4).unwrap();
        let second_move = Loc::new(4, 4).unwrap();

        let first_index = parent.add_child(first_move, 0.6, first_ptr);
        let second_index = parent.add_child(second_move, 0.4, second_ptr);

        assert_eq!(first_index, 0);
        assert_eq!(second_index, 1);
        assert_eq!(parent.children.0[first_index].move_loc, first_move);
        assert_eq!(parent.children.0[first_index].visits, 0);
        assert_eq!(parent.children.0[first_index].policy_prior, 0.6);
        assert_eq!(parent.children.0[first_index].child, first_ptr);
        assert_eq!(parent.children.0[second_index].move_loc, second_move);
        assert_eq!(parent.children.0[second_index].child, second_ptr);
    }
}
