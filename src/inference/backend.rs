use crate::inference::{outputs::RawSearchNNOutputs, policy::POLICY_SIZE};

use super::inputs::NNInputs;

#[derive(Debug)]
pub enum InferenceError {
    ExecutionFailed,
}
pub trait InferenceBackend {
    /// Replaces `outputs` with one result per input, preserving input order.
    fn evaluate_batch(
        &mut self,
        inputs: &[NNInputs],
        outputs: &mut Vec<RawSearchNNOutputs>,
    ) -> Result<(), InferenceError>;
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct DummyInferenceBackend {
    pub(crate) batch_sizes: Vec<usize>, // for debugging
}

#[cfg(test)]
impl InferenceBackend for DummyInferenceBackend {
    fn evaluate_batch(
        &mut self,
        inputs: &[NNInputs],
        outputs: &mut Vec<RawSearchNNOutputs>,
    ) -> Result<(), InferenceError> {
        outputs.clear();
        outputs.reserve(inputs.len()); //should be noop if you are keeping consistent batch sizes, safeguard
        for _input in inputs {
            outputs.push(RawSearchNNOutputs::new([0.0; POLICY_SIZE], 0.0, 0.0, 0.0));
        }
        self.batch_sizes.push(inputs.len());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::game_state::GameState;
    use crate::game::rules::Rules;

    #[test]
    fn dummy_backend_replaces_outputs_and_records_batch_sizes() {
        let game_state = GameState::new(Rules::TROMP_TAYLORISH);
        let inputs: [NNInputs; 3] = std::array::from_fn(|_| NNInputs::encode(&game_state));
        let mut backend = DummyInferenceBackend::default();
        let mut outputs = Vec::new();

        backend
            .evaluate_batch(&inputs, &mut outputs)
            .expect("first dummy batch should succeed");
        assert_eq!(outputs.len(), 3);

        backend
            .evaluate_batch(&inputs[..1], &mut outputs)
            .expect("second dummy batch should succeed");
        assert_eq!(outputs.len(), 1);
        assert_eq!(backend.batch_sizes, [3, 1]);
    }
}
