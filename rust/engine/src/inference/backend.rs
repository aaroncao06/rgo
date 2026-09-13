use std::sync::Arc;

use crate::inference::outputs::NNOutput;

#[cfg(test)]
use crate::inference::policy::POLICY_SIZE;

use super::inputs::NNInput;

#[derive(Debug, Clone)]
pub(crate) enum InferenceError {
    ExecutionFailed,
    MismatchedBatchOutput,
    RuntimeClosed,
}
pub(crate) trait InferenceBackend {
    /// appends to outputs one result per input, preserving input order.
    /// on success, outputs.len must equal inputs.len, else returns err
    fn evaluate_batch(
        &mut self,
        inputs: &[NNInput],
        outputs: &mut Vec<Arc<NNOutput>>,
    ) -> Result<(), InferenceError>;
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
        inputs: &[NNInput],
        outputs: &mut Vec<Arc<NNOutput>>,
    ) -> Result<(), InferenceError> {
        debug_assert!(outputs.is_empty()); // up to the executor to clear before calling
        // outputs.reserve(inputs.len()); //should be noop if you are keeping consistent batch sizes, safeguard
        for _input in inputs {
            outputs.push(Arc::new(NNOutput::from_raw(
                [0.0; POLICY_SIZE],
                0.0,
                0.0,
                0.0,
            )));
        }
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
        let game_state = GameState::new(Rules::TROMP_TAYLORISH);
        let inputs: [NNInput; 3] = std::array::from_fn(|_| NNInput::encode(&game_state));
        let mut backend = DummyInferenceBackend::default();
        let mut outputs = Vec::new();

        backend
            .evaluate_batch(&inputs, &mut outputs)
            .expect("first dummy batch should succeed");
        assert_eq!(outputs.len(), 3);

        outputs.clear();
        backend
            .evaluate_batch(&inputs[..1], &mut outputs)
            .expect("second dummy batch should succeed");
        assert_eq!(outputs.len(), 1);
        assert_eq!(backend.batch_sizes, [3, 1]);
    }
}
