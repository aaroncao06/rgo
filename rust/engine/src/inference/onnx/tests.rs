use super::*;
use crate::game::board::Player;

fn backend() -> OnnxBackend {
    OnnxBackend::load(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/v0.onnx"),
        1,
    )
    .unwrap()
}

#[test]
fn batches_raw_outputs_and_attaches_only_requested_ownership() {
    let inputs: Vec<_> = (0..3)
        .map(|i| NNInput {
            spatial: [i as f32; NUM_SPATIAL_FEATURES * BOARD_POLICY_SIZE],
            global: [i as f32, i as f32],
            include_ownership: i == 1,
        })
        .collect();
    let mut backend = backend();
    let mut outputs = Vec::new();
    backend.evaluate_batch(&inputs, &mut outputs).unwrap();
    assert_eq!(outputs.len(), inputs.len());
    for (i, output) in outputs.iter_mut().enumerate() {
        assert!(!output.is_processed());
        assert_eq!(output.has_ownership(), i == 1);
        let output = Arc::get_mut(output).unwrap();
        output.process_in_place(Player::White, &[true; POLICY_SIZE]);
        assert!((output.white_score_mean() - i as f32 * 20.0).abs() < 1e-6);
        if let Some(ownership) = output.white_ownership() {
            assert!((ownership[0] - (i as f32).tanh()).abs() < 1e-6);
        }
    }
    // Reuse packing buffers with a different batch size and no ownership fetch.
    outputs.clear();
    backend.evaluate_batch(&inputs[..1], &mut outputs).unwrap();
    assert_eq!(outputs.len(), 1);
    assert!(!outputs[0].has_ownership());
}

#[cfg(debug_assertions)]
#[test]
fn debug_checks_reject_nonfinite_model_outputs() {
    let mut backend = backend();
    let mut outputs = Vec::new();
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for invalid_policy in [true, false] {
            let input = NNInput {
                spatial: [if invalid_policy { invalid } else { 0.0 };
                    NUM_SPATIAL_FEATURES * BOARD_POLICY_SIZE],
                global: [if invalid_policy { 0.0 } else { invalid }; NUM_GLOBAL_FEATURES],
                include_ownership: true,
            };
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                backend.evaluate_batch(&[input], &mut outputs)
            }));
            assert!(panic.is_err());
            assert!(outputs.is_empty());
        }
    }
}

#[test]
fn rejects_incompatible_contract_metadata() {
    let session = Session::builder()
        .unwrap()
        .commit_from_memory(include_bytes!("../../../tests/data/wrong_version.onnx"))
        .unwrap();
    assert!(validate_contract(&session).is_err());
}

#[test]
fn rejects_incompatible_input_shape() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/wrong_shape.onnx");
    assert!(OnnxBackend::load(&path, 1).is_err());
}

#[test]
fn preserves_policy_order_including_pass() {
    let mut backend = backend();
    let input = NNInput {
        spatial: std::array::from_fn(|i| i as f32 / 100.0),
        global: [0.0; NUM_GLOBAL_FEATURES],
        include_ownership: false,
    };
    let mut outputs = Vec::new();
    backend.evaluate_batch(&[input], &mut outputs).unwrap();
    let output = Arc::get_mut(&mut outputs[0]).unwrap();
    output.process_in_place(Player::White, &[true; POLICY_SIZE]);
    let probabilities = output.policy_probs();
    for i in 0..POLICY_SIZE {
        assert!((probabilities[i] / probabilities[0] - (i as f32 / 100.0).exp()).abs() < 1e-5);
    }
}
