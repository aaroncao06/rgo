use crate::{game::board::Loc, inference::outputs::NNOutput};
use std::{ptr::NonNull, sync::Arc};
pub struct SearchNode {
    nn_output: Option<Arc<NNOutput>>,
    stats: SearchStats,
    children: ChildStorage,
}

/// Accumulated weighted values and visit counts for a search node.
#[derive(Debug, Default)]
pub(super) struct SearchStats {
    pub(super) visits: i32,
    pub(super) white_win_sum: f64, //higher precision
    pub(super) white_score_sum: f64,
    pub(super) white_score_mean_sq_sum: f64,
    pub(super) white_utility_sum: f64,
    pub(super) white_utility_sq_sum: f64,
    pub(super) weight_sum: f64,
    pub(super) weight_sq_sum: f64,
}

struct ChildStorage(Vec<Edge>); //naive first version, postpone staged storage opt

/// Opaque handle to an edge. Keeping the storage position behind this type lets
/// ChildStorage change from a flat Vec to staged or split storage without
/// exposing that representation to traversal and backup code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct EdgeIndex(usize);

pub(super) struct Edge {
    move_loc: Loc,
    visits: i32,       // used for graph search
    policy_prior: f32, //initial policy prob
    child: NonNull<SearchNode>,
}

impl Edge {
    pub(super) fn move_loc(&self) -> Loc {
        self.move_loc
    }
    pub(super) fn visits(&self) -> i32 {
        self.visits
    }
    pub(super) fn policy_prior(&self) -> f32 {
        self.policy_prior
    }
    pub(super) fn child(&self) -> NonNull<SearchNode> {
        self.child
    }
    /// # Safety
    /// The child pointer must reference a live, initialized node that is not
    /// mutably borrowed for the duration of this call.
    pub(super) unsafe fn child_weight(&self) -> f64 {
        // how much of the child's weight belongs to this edge
        let child = unsafe { self.child.as_ref() };
        child.weight_sum() * f64::from(self.visits) / f64::from(child.visits().max(1))
    }
    pub(super) fn record_visit(&mut self) {
        self.visits += 1;
    }
}

impl SearchNode {
    pub(super) fn new() -> Self {
        Self {
            nn_output: None,
            stats: SearchStats::default(),
            children: ChildStorage::new(),
        }
    }
    pub(super) fn initialize_from_nn_eval(&mut self, nn_output: Arc<NNOutput>, white_utility: f64) {
        debug_assert!(self.nn_output.is_none(), "only populate empty nodes");
        debug_assert_eq!(self.stats.visits, 0, "only initialize unvisited nodes");
        debug_assert!(nn_output.is_processed());
        let white_win = f64::from(nn_output.white_win_prob());
        let white_score = f64::from(nn_output.white_score_mean());
        let white_score_mean_sq = f64::from(nn_output.white_score_mean_sq());
        self.nn_output = Some(nn_output);
        self.record_visit(white_win, white_score, white_score_mean_sq, white_utility);
    }
    pub(super) fn record_visit(
        &mut self,
        white_win: f64,
        white_score: f64,
        white_score_mean_sq: f64,
        white_utility: f64,
    ) {
        self.stats.visits += 1;
        self.stats.white_win_sum += white_win;
        self.stats.white_score_sum += white_score;
        self.stats.white_score_mean_sq_sum += white_score_mean_sq;
        self.stats.white_utility_sum += white_utility;
        self.stats.white_utility_sq_sum += white_utility * white_utility;
        self.stats.weight_sum += 1.0;
        self.stats.weight_sq_sum += 1.0;
    }
    pub(super) fn repeat_visit(&mut self) {
        let white_win = self.white_win_rate();
        let white_score = self.white_score_mean();
        let white_score_mean_sq = self.white_score_mean_sq();
        let white_utility = self.white_utility();
        self.record_visit(white_win, white_score, white_score_mean_sq, white_utility);
    }
    pub(super) fn policy_probs(&self) -> &[f32; crate::inference::policy::POLICY_SIZE] {
        self.nn_output
            .as_ref()
            .expect("search nodes must be evaluated before traversal")
            .policy_probs()
    }
    /// Prepare the root policy before creating children. Clones the inference
    /// output if it is shared, so cached outputs remain unchanged. Existing
    /// edges retain their own priors and would not reflect later policy edits.
    pub(super) fn policy_probs_mut(&mut self) -> &mut [f32; crate::inference::policy::POLICY_SIZE] {
        std::sync::Arc::make_mut(
            self.nn_output
                .as_mut()
                .expect("search nodes must be evaluated before traversal"),
        )
        .policy_probs_mut()
    }
    pub(super) fn nn_output(&self) -> &NNOutput {
        self.nn_output
            .as_deref()
            .expect("search nodes must be evaluated before traversal")
    }
    pub(super) fn edge_visit_sum(&self) -> i32 {
        self.children.iter().map(|edge| edge.visits).sum()
    }
    /// # Safety
    /// Every child must satisfy `Edge::child_weight`'s safety requirements.
    pub(super) unsafe fn child_weight_sum(&self) -> f64 {
        self.children
            .iter()
            .map(|edge| unsafe { edge.child_weight() })
            .sum()
    }
    pub(super) fn edges(&self) -> impl Iterator<Item = &Edge> {
        self.children.iter()
    }
    pub(super) fn indexed_edges(&self) -> impl Iterator<Item = (EdgeIndex, &Edge)> {
        self.children.indexed_iter()
    }
    pub(super) fn edge(&self, edge_index: EdgeIndex) -> &Edge {
        self.children.get(edge_index)
    }
    pub(super) fn visited_policy_mass(&self) -> f64 {
        self.children
            .iter()
            .map(|edge| f64::from(edge.policy_prior))
            .sum()
    }
    pub(super) fn edge_for_move(&self, move_loc: Loc) -> Option<(EdgeIndex, &Edge)> {
        self.children.edge_for_move(move_loc)
    }
    pub(super) fn edge_mut(&mut self, edge_index: EdgeIndex) -> &mut Edge {
        self.children.get_mut(edge_index)
    }
    pub(super) fn visits(&self) -> i32 {
        self.stats.visits
    }
    pub(super) fn white_win_rate(&self) -> f64 {
        debug_assert!(self.stats.weight_sum > 0.0);
        self.stats.white_win_sum / self.stats.weight_sum
    }
    pub(super) fn white_score_mean(&self) -> f64 {
        debug_assert!(self.stats.weight_sum > 0.0);
        self.stats.white_score_sum / self.stats.weight_sum
    }
    pub(super) fn white_score_mean_sq(&self) -> f64 {
        debug_assert!(self.stats.weight_sum > 0.0);
        self.stats.white_score_mean_sq_sum / self.stats.weight_sum
    }
    pub(super) fn white_utility(&self) -> f64 {
        debug_assert!(self.stats.weight_sum > 0.0);
        self.stats.white_utility_sum / self.stats.weight_sum
    }
    pub(super) fn white_utility_mean_sq(&self) -> f64 {
        debug_assert!(self.stats.weight_sum > 0.0);
        self.stats.white_utility_sq_sum / self.stats.weight_sum
    }
    pub(super) fn weight_sum(&self) -> f64 {
        self.stats.weight_sum
    }
    pub(super) fn weight_sq_sum(&self) -> f64 {
        self.stats.weight_sq_sum
    }
    pub(super) fn replace_stats(&mut self, stats: SearchStats) {
        debug_assert!(stats.visits > 0);
        debug_assert!(stats.weight_sum > 0.0);
        self.stats = stats;
    }
    pub(super) fn add_child(
        &mut self,
        move_loc: Loc,
        policy_prior: f32,
        child: NonNull<SearchNode>,
    ) -> EdgeIndex {
        debug_assert!(policy_prior.is_finite());
        debug_assert!((0.0..=1.0).contains(&policy_prior));

        self.children.insert(Edge {
            move_loc,
            visits: 0, //visits get incremented during backup
            policy_prior,
            child,
        })
    }
}
impl ChildStorage {
    fn new() -> Self {
        Self(Vec::new())
    }
    fn iter(&self) -> impl Iterator<Item = &Edge> {
        self.0.iter()
    }
    fn indexed_iter(&self) -> impl Iterator<Item = (EdgeIndex, &Edge)> {
        self.0
            .iter()
            .enumerate()
            .map(|(index, edge)| (EdgeIndex(index), edge))
    }
    fn edge_for_move(&self, move_loc: Loc) -> Option<(EdgeIndex, &Edge)> {
        self.indexed_iter()
            .find(|(_, edge)| edge.move_loc == move_loc)
    }
    fn insert(&mut self, edge: Edge) -> EdgeIndex {
        let index = EdgeIndex(self.0.len());
        self.0.push(edge);
        index
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

    fn loc(x: usize, y: usize) -> Loc {
        crate::game::board::Board::new(9).loc(x, y).unwrap()
    }

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
        assert_eq!(node.stats.visits, 0);
        assert_eq!(node.stats.white_win_sum, 0.0);
        assert_eq!(node.stats.white_score_sum, 0.0);
        assert_eq!(node.stats.white_score_mean_sq_sum, 0.0);
        assert_eq!(node.stats.white_utility_sum, 0.0);
        assert_eq!(node.stats.white_utility_sq_sum, 0.0);
        assert!(node.edges().next().is_none());
    }

    #[test]
    fn nn_initialization_records_the_first_visit() {
        let mut node = SearchNode::new();
        let output = processed_output();
        let initial_win = f64::from(output.white_win_prob());
        let initial_score = f64::from(output.white_score_mean());
        let initial_score_mean_sq = f64::from(output.white_score_mean_sq());

        node.initialize_from_nn_eval(output.clone(), 0.6);
        node.record_visit(0.25, -1.5, 6.0, -0.2);

        assert!(Arc::ptr_eq(node.nn_output.as_ref().unwrap(), &output));
        assert_eq!(node.stats.visits, 2);
        assert_eq!(node.stats.white_win_sum, initial_win + 0.25);
        assert_eq!(node.stats.white_score_sum, initial_score - 1.5);
        assert_eq!(
            node.stats.white_score_mean_sq_sum,
            initial_score_mean_sq + 6.0
        );
        assert!((node.stats.white_utility_sum - 0.4).abs() < 1e-12);
        assert!((node.stats.white_utility_sq_sum - 0.4).abs() < 1e-12);
        assert_eq!(node.visits(), 2);
        assert_eq!(node.white_win_rate(), (initial_win + 0.25) / 2.0);
        assert_eq!(node.white_score_mean(), (initial_score - 1.5) / 2.0);
        assert_eq!(
            node.white_score_mean_sq(),
            (initial_score_mean_sq + 6.0) / 2.0
        );
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
        let first_move = loc(3, 4);
        let second_move = loc(4, 4);

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
