use std::{ptr::NonNull, sync::Arc};

use crate::{
    game::{board::Loc, game_state::GameState},
    inference::{backend::InferenceError, outputs::NNOutput, runtime::InferenceClient},
    search::{
        graph_key::GraphKey,
        node::{EdgeIndex, SearchNode},
        node_store::{NodeStore, StoreFull},
    },
};

struct SearchRoot {
    game_state: GameState, // playouts will start from this
    key: GraphKey,
    node: Box<SearchNode>, // owned separately from the rest of the nodestore
}

//search state tracks traversal through mcts tree. reset in between moves, not playouts. eventually will support pruning
//owned by search worker
pub(crate) struct SearchGraph<N: NodeStore> {
    root: Option<SearchRoot>,
    node_store: N,
}

impl<N: NodeStore> SearchGraph<N> {
    pub(crate) fn new(node_store: N) -> Self {
        Self {
            root: None,
            node_store,
        }
    }
    pub(crate) fn reset(&mut self, root_game_state: GameState, root_output: Arc<NNOutput>) {
        self.node_store.clear();
        let key = GraphKey::new(&root_game_state);
        let node = Self::initialized_root(root_output);
        self.root = Some(SearchRoot {
            game_state: root_game_state,
            key,
            node,
        });
    }
    fn initialized_root(output: Arc<NNOutput>) -> Box<SearchNode> {
        let mut root = Box::new(SearchNode::new());
        root.attach_nn_output(output.clone());
        root.record_visit(
            f64::from(output.white_win_prob()),
            f64::from(output.white_score_mean()),
            f64::from(output.white_score_mean_sq()),
        );
        root
    }
    pub(crate) fn advance_root(&mut self, move_loc: Loc) {
        // later for normal play
    }
}

struct PlayoutStep {
    parent: NonNull<SearchNode>,
    edge_index: EdgeIndex,
}

struct PlayoutPath {
    steps: Vec<PlayoutStep>,
}
impl PlayoutPath {
    fn new() -> Self {
        Self { steps: Vec::new() }
    }
    fn clear(&mut self) {
        self.steps.clear();
    }
    fn push(&mut self, parent: NonNull<SearchNode>, edge_index: EdgeIndex) {
        self.steps.push(PlayoutStep { parent, edge_index });
    }
}

struct SearchWorker<N: NodeStore> {
    // keeps mutating its search state, one worker per game, eventually supports subgraph reuse
    search_graph: SearchGraph<N>,
    playout_path: PlayoutPath, // scratch work to avoid reallocating
    scratch_game_state: Option<GameState>, // mutates through each playout
}

#[derive(Debug)]
pub(crate) enum SearchError {
    StoreFull,
    InferenceError(InferenceError),
}

impl From<InferenceError> for SearchError {
    fn from(error: InferenceError) -> Self {
        Self::InferenceError(error)
    }
}

impl From<StoreFull> for SearchError {
    fn from(_: StoreFull) -> Self {
        Self::StoreFull
    }
}

impl<N: NodeStore> SearchWorker<N> {
    fn new(node_store: N) -> Self {
        Self {
            search_graph: SearchGraph::new(node_store),
            playout_path: PlayoutPath::new(),
            scratch_game_state: None,
        }
    }
    async fn start_game(
        &mut self,
        root_game_state: GameState,
        inference_client: &mut InferenceClient,
    ) -> Result<(), SearchError> {
        // just so that cross game graph reuse is not allowed
        self.reset_graph(root_game_state, inference_client).await
    }
    async fn reset_graph(
        &mut self,
        root_game_state: GameState,
        inference_client: &mut InferenceClient,
    ) -> Result<(), SearchError> {
        // full reset, in the future can have a version where you retain subgraph between moves
        debug_assert!(!root_game_state.is_finished());
        let root_output = inference_client.evaluate(&root_game_state).await?;

        match self.scratch_game_state.as_mut() {
            Some(scratch) => scratch.reset_from(&root_game_state),
            None => self.scratch_game_state = Some(root_game_state.clone()),
        }
        self.search_graph.reset(root_game_state, root_output);
        self.playout_path.clear();
        Ok(())
    }
    async fn playout(&mut self, inference_client: &mut InferenceClient) -> Result<(), SearchError> {
        self.playout_path.clear();
        let root = self
            .search_graph
            .root
            .as_mut()
            .expect("playout requires an active game");
        let scratch_game_state = self
            .scratch_game_state
            .as_mut()
            .expect("active game requires scratch state");
        scratch_game_state.reset_from(&root.game_state);
        let mut current_node = NonNull::from(root.node.as_mut());

        loop {
            if self
                .scratch_game_state
                .as_ref()
                .expect("active game requires scratch state")
                .is_finished()
            {
                break;
            }
            let node = unsafe { current_node.as_ref() };

            // choose move, check if move is in children. if so go down, if not break
        }

        Ok(())
    }
    fn choose_move(&self, node: &SearchNode, game_state: &GameState) -> Result<Loc, SearchError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        game::rules::Rules,
        inference::{
            backend::InferenceBackend, inputs::NNInput, policy::POLICY_SIZE, runtime::ModelRuntime,
        },
        search::node_store::FixedArenaNodeStore,
    };

    struct TestBackend {
        fail: bool,
    }

    impl InferenceBackend for TestBackend {
        fn evaluate_batch(
            &mut self,
            inputs: &[NNInput],
            outputs: &mut Vec<Arc<NNOutput>>,
        ) -> Result<(), InferenceError> {
            if self.fail {
                return Err(InferenceError::ExecutionFailed);
            }
            for _ in inputs {
                outputs.push(Arc::new(NNOutput::from_raw(
                    [0.0; POLICY_SIZE],
                    0.0,
                    0.0,
                    0.0,
                )));
            }
            Ok(())
        }
    }

    fn inference_client(fail: bool) -> InferenceClient {
        let model_handle = ModelRuntime::start(vec![TestBackend { fail }], 1, 1, 16, 1);
        InferenceClient::new(model_handle)
    }

    fn worker() -> SearchWorker<FixedArenaNodeStore> {
        SearchWorker::new(FixedArenaNodeStore::new(16))
    }

    #[test]
    fn new_worker_has_no_active_graph_or_scratch_state() {
        let worker = worker();

        assert!(worker.search_graph.root.is_none());
        assert_eq!(worker.search_graph.node_store.len(), 0);
        assert!(worker.scratch_game_state.is_none());
        assert!(worker.playout_path.steps.is_empty());
    }

    #[tokio::test]
    async fn start_game_installs_an_evaluated_root_and_scratch_state() {
        let mut worker = worker();
        let mut client = inference_client(false);
        let game_state = GameState::new(Rules::TROMP_TAYLORISH);
        let expected_key = GraphKey::new(&game_state);

        worker.start_game(game_state, &mut client).await.unwrap();

        let root = worker.search_graph.root.as_ref().unwrap();
        assert_eq!(root.key, expected_key);
        assert_eq!(root.node.visits(), 1);
        assert_eq!(
            GraphKey::new(worker.scratch_game_state.as_ref().unwrap()),
            expected_key
        );
    }

    #[tokio::test]
    async fn resetting_graph_clears_stored_nodes_and_replaces_the_root() {
        let mut worker = worker();
        let mut client = inference_client(false);
        worker
            .start_game(GameState::new(Rules::TROMP_TAYLORISH), &mut client)
            .await
            .unwrap();
        worker
            .search_graph
            .node_store
            .find_or_insert(GraphKey::from_raw(1))
            .unwrap();

        let mut next_state = GameState::new(Rules::TROMP_TAYLORISH);
        assert!(next_state.play(Loc::new(4, 4).unwrap()));
        let expected_key = GraphKey::new(&next_state);
        worker.reset_graph(next_state, &mut client).await.unwrap();

        assert_eq!(worker.search_graph.node_store.len(), 0);
        assert_eq!(worker.search_graph.root.as_ref().unwrap().key, expected_key);
    }

    #[tokio::test]
    async fn inference_failure_preserves_the_existing_graph() {
        let mut worker = worker();
        let mut working_client = inference_client(false);
        worker
            .start_game(GameState::new(Rules::TROMP_TAYLORISH), &mut working_client)
            .await
            .unwrap();
        worker
            .search_graph
            .node_store
            .find_or_insert(GraphKey::from_raw(1))
            .unwrap();
        let original_key = worker.search_graph.root.as_ref().unwrap().key;

        let mut failing_client = inference_client(true);
        let mut replacement = GameState::new(Rules::TROMP_TAYLORISH);
        assert!(replacement.play(Loc::new(4, 4).unwrap()));
        let result = worker.reset_graph(replacement, &mut failing_client).await;

        assert!(matches!(result, Err(SearchError::InferenceError(_))));
        assert_eq!(worker.search_graph.root.as_ref().unwrap().key, original_key);
        assert_eq!(worker.search_graph.node_store.len(), 1);
    }
}
