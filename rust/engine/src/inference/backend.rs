use std::sync::Arc;

use crate::inference::outputs::NNOutput;

use super::inputs::NNInput;

/// A homogeneous batch borrowed from request storage. Each visit finishes
/// before the next begins; implementations must not retain input references.
pub trait InputBatch {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn board_dim(&self) -> usize;
    fn for_each_input(&self, visit: &mut dyn FnMut(&NNInput));
}

#[derive(Debug, Clone)]
pub enum InferenceError {
    ExecutionFailed,
    MismatchedBatchOutput,
    RuntimeClosed,
    UnsupportedBoardDim(usize),
    Onnx(Arc<ort::Error>),
}
pub trait InferenceBackend {
    /// Evaluate raw model activations in input order.
    ///
    /// The executor supplies a nonempty input batch and an empty, reusable
    /// output vector. Every input in a batch has the same active board size.
    /// Gather input data during `for_each_input` visits, then execute the model
    /// after visits return so request storage is not locked during inference.
    /// On success,
    /// append exactly one output per input; return an error if the backend
    /// cannot produce that complete batch. Partial outputs on error are ignored.
    /// Returned raw activations must be finite. This is a model/backend contract
    /// invariant checked with debug assertions, not a release-time tensor scan.
    /// Each output must be unprocessed and exclusively owned: do not retain
    /// other strong or weak Arc references or share one output between rows.
    /// Policy contains exactly board_dim² active row-major logits, then pass.
    /// The client uses Arc::get_mut to apply legal masking and perspective/score
    /// transformations before sharing the result with the model cache and search.
    /// When input.include_ownership is true, attach current-player ownership
    /// logits as an exclusively owned, exact-size Arc plane of board_dim² values
    /// via with_ownership_logits. Other requests may omit that output.
    fn evaluate_batch(
        &mut self,
        inputs: &dyn InputBatch,
        outputs: &mut Vec<Arc<NNOutput>>,
    ) -> Result<(), InferenceError>;
}

#[cfg(test)]
impl<T: AsRef<[NNInput]>> InputBatch for T {
    fn len(&self) -> usize {
        self.as_ref().len()
    }
    fn board_dim(&self) -> usize {
        self.as_ref()[0].board_dim
    }
    fn for_each_input(&self, visit: &mut dyn FnMut(&NNInput)) {
        for input in self.as_ref() {
            visit(input);
        }
    }
}

#[cfg(test)]
#[derive(Default)]
struct DummyInferenceBackend {
    batch_sizes: Vec<usize>, // for debugging
}

#[cfg(test)]
impl InferenceBackend for DummyInferenceBackend {
    fn evaluate_batch(
        &mut self,
        inputs: &dyn InputBatch,
        outputs: &mut Vec<Arc<NNOutput>>,
    ) -> Result<(), InferenceError> {
        debug_assert!(outputs.is_empty()); // up to the executor to clear before calling
        // outputs.reserve(inputs.len()); //should be noop if you are keeping consistent batch sizes, safeguard
        inputs.for_each_input(&mut |input| {
            let mut output = NNOutput::from_raw(
                vec![0.0; input.board_dim * input.board_dim + 1].into_boxed_slice(),
                0.0,
                0.0,
                0.0,
            );
            if input.include_ownership {
                output = output
                    .with_ownership_logits(vec![0.0; input.board_dim * input.board_dim].into());
            }
            outputs.push(Arc::new(output));
        });
        self.batch_sizes.push(inputs.len());
        if inputs.len() != outputs.len() {
            return Err(InferenceError::MismatchedBatchOutput);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::game_state::GameState;
    use crate::game::rules::Rules;

    #[test]
    fn dummy_backend_populates_empty_outputs_and_records_batch_sizes() {
        let game_state = GameState::new(Rules::TROMP_TAYLORISH_9);
        let inputs: [NNInput; 3] = std::array::from_fn(|_| NNInput::encode(&game_state));
        let mut backend = DummyInferenceBackend::default();
        let mut outputs = Vec::new();

        backend
            .evaluate_batch(&inputs, &mut outputs)
            .expect("first dummy batch should succeed");
        assert_eq!(outputs.len(), 3);

        outputs.clear();
        backend
            .evaluate_batch(&&inputs[..1], &mut outputs)
            .expect("second dummy batch should succeed");
        assert_eq!(outputs.len(), 1);
        assert_eq!(backend.batch_sizes, [3, 1]);
    }
}
