use crate::{game::board::Loc, inference::outputs::NNOutput};
use std::{ptr::NonNull, sync::Arc};
pub(crate) struct SearchNode {
    nn_output: Option<Arc<NNOutput>>,
    visits: i32,
    white_win_sum: f64, //higher precision
    white_score_sum: f64,
    white_score_mean_sq_sum: f64,
    white_utility_sum: f64,
    white_utility_sq_sum: f64,
    children: ChildStorage,
}

struct ChildStorage(Vec<Edge>); //naive first version, postpone staged storage opt

/// Opaque handle to an edge. Keeping the storage position behind this type lets
/// ChildStorage change from a flat Vec to staged or split storage without
/// exposing that representation to traversal and backup code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EdgeIndex(usize);

pub(crate) struct Edge {
    move_loc: Loc,
    visits: i32,       // used for graph search
    policy_prior: f32, //initial policy prob
    child: NonNull<SearchNode>,
}

impl Edge {
    pub(crate) fn move_loc(&self) -> Loc {
        self.move_loc
    }
    pub(crate) fn visits(&self) -> i32 {
        self.visits
    }
    pub(crate) fn policy_prior(&self) -> f32 {
        self.policy_prior
    }
    pub(crate) fn child(&self) -> NonNull<SearchNode> {
        self.child
    }
    pub(crate) fn record_visit(&mut self) {
        self.visits += 1;
    }
}

impl SearchNode {
    pub(crate) fn new() -> Self {
        Self {
            nn_output: None,
            visits: 0,
            white_win_sum: 0.0,
            white_score_sum: 0.0,
            white_score_mean_sq_sum: 0.0,
            white_utility_sum: 0.0,
            white_utility_sq_sum: 0.0,
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
        white_utility: f64,
    ) {
        self.visits += 1;
        self.white_win_sum += white_win;
        self.white_score_sum += white_score;
        self.white_score_mean_sq_sum += white_score_mean_sq;
        self.white_utility_sum += white_utility;
        self.white_utility_sq_sum += white_utility * white_utility;
    }
    pub(crate) fn policy_probs(&self) -> &[f32; crate::inference::policy::POLICY_SIZE] {
        self.nn_output
            .as_ref()
            .expect("search nodes must be evaluated before traversal")
            .policy_probs()
    }
    pub(crate) fn nn_output(&self) -> &NNOutput {
        self.nn_output
            .as_deref()
            .expect("search nodes must be evaluated before traversal")
    }
    pub(crate) fn edge_visit_sum(&self) -> i32 {
        self.children.0.iter().map(|edge| edge.visits).sum()
    }
    pub(crate) fn visited_policy_mass(&self) -> f64 {
        self.children
            .0
            .iter()
            .map(|edge| f64::from(edge.policy_prior))
            .sum()
    }
    pub(crate) fn edge_for_move(&self, move_loc: Loc) -> Option<(EdgeIndex, &Edge)> {
        self.children
            .0
            .iter()
            .enumerate()
            .find(|(_, edge)| edge.move_loc == move_loc)
            .map(|(index, edge)| (EdgeIndex(index), edge))
    }
    pub(crate) fn edge_mut(&mut self, edge_index: EdgeIndex) -> &mut Edge {
        self.children.get_mut(edge_index)
    }
    pub(crate) fn visits(&self) -> i32 {
        self.visits
    }
    pub(crate) fn white_win_rate(&self) -> f64 {
        debug_assert!(self.visits > 0);
        self.white_win_sum / f64::from(self.visits)
    }
    pub(crate) fn white_score_mean(&self) -> f64 {
        debug_assert!(self.visits > 0);
        self.white_score_sum / f64::from(self.visits)
    }
    pub(crate) fn white_score_mean_sq(&self) -> f64 {
        debug_assert!(self.visits > 0);
        self.white_score_mean_sq_sum / f64::from(self.visits)
    }
    pub(crate) fn white_utility(&self) -> f64 {
        debug_assert!(self.visits > 0);
        self.white_utility_sum / f64::from(self.visits)
    }
    pub(crate) fn white_utility_mean_sq(&self) -> f64 {
        debug_assert!(self.visits > 0);
        self.white_utility_sq_sum / f64::from(self.visits)
    }
    pub(crate) fn add_child(
        &mut self,
        move_loc: Loc,
        policy_prior: f32,
        child: NonNull<SearchNode>,
    ) -> EdgeIndex {
        debug_assert!(policy_prior.is_finite());
        debug_assert!((0.0..=1.0).contains(&policy_prior));

        let edge_index = self.children.0.len();

        self.children.0.push(Edge {
            move_loc,
            visits: 0, //visits get incremented during backup
            policy_prior,
            child,
        });

        EdgeIndex(edge_index)
    }
}
impl ChildStorage {
    fn new() -> Self {
        Self(Vec::new())
    }
    fn get(&self, edge_index: EdgeIndex) -> &Edge {
        &self.0[edge_index.0]
    }
    fn get_mut(&mut self, edge_index: EdgeIndex) -> &mut Edge {
        &mut self.0[edge_index.0]
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
        assert_eq!(node.white_utility_sum, 0.0);
        assert_eq!(node.white_utility_sq_sum, 0.0);
        assert!(node.children.0.is_empty());
    }

    #[test]
    fn output_attachment_and_visit_recording_update_separate_state() {
        let mut node = SearchNode::new();
        let output = processed_output();

        node.attach_nn_output(output.clone());
        node.record_visit(0.75, 3.5, 14.0, 0.6);
        node.record_visit(0.25, -1.5, 6.0, -0.2);

        assert!(Arc::ptr_eq(node.nn_output.as_ref().unwrap(), &output));
        assert_eq!(node.visits, 2);
        assert_eq!(node.white_win_sum, 1.0);
        assert_eq!(node.white_score_sum, 2.0);
        assert_eq!(node.white_score_mean_sq_sum, 20.0);
        assert!((node.white_utility_sum - 0.4).abs() < 1e-12);
        assert!((node.white_utility_sq_sum - 0.4).abs() < 1e-12);
        assert_eq!(node.visits(), 2);
        assert_eq!(node.white_win_rate(), 0.5);
        assert_eq!(node.white_score_mean(), 1.0);
        assert_eq!(node.white_score_mean_sq(), 10.0);
        assert!((node.white_utility() - 0.2).abs() < 1e-12);
        assert!((node.white_utility_mean_sq() - 0.2).abs() < 1e-12);
        assert_eq!(node.policy_probs(), output.policy_probs());
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

        assert_eq!(first_index, EdgeIndex(0));
        assert_eq!(second_index, EdgeIndex(1));
        assert_eq!(parent.children.get(first_index).move_loc(), first_move);
        assert_eq!(parent.children.get(first_index).visits(), 0);
        assert_eq!(parent.children.get(first_index).policy_prior(), 0.6);
        assert_eq!(parent.children.get(first_index).child(), first_ptr);
        assert_eq!(parent.children.get(second_index).move_loc(), second_move);
        assert_eq!(parent.children.get(second_index).child(), second_ptr);

        assert_eq!(parent.edge_visit_sum(), 0);
        let (found_index, first_edge) = parent.edge_for_move(first_move).unwrap();
        assert_eq!(found_index, first_index);
        assert_eq!(first_edge.visits(), 0);
        assert_eq!(first_edge.policy_prior(), 0.6);
        assert_eq!(first_edge.child(), first_ptr);
        assert!(parent.edge_for_move(Loc::PASS).is_none());

        parent.edge_mut(first_index).record_visit();
        assert_eq!(parent.edge_for_move(first_move).unwrap().1.visits(), 1);
        assert_eq!(parent.edge_visit_sum(), 1);
    }
}
