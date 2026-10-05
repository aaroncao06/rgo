use super::*;
use crate::game::board::{MAX_BOARD_AREA, Player};
use crate::inference::inputs::NNInput;
use crate::inference::policy::MAX_POLICY_SIZE;

fn cpu_config() -> ExecutorConfig {
    ExecutorConfig {
        device: InferenceDevice::Cpu {},
        intra_threads: 1,
        base_batch_size: 8,
    }
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
struct CacheDir(PathBuf);

#[cfg(all(feature = "coreml", target_os = "macos"))]
impl CacheDir {
    fn new(provider: &str) -> Self {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = Self(std::env::temp_dir().join(format!(
            "rgo-{provider}-cache-{}-{unique}",
            std::process::id()
        )));
        std::fs::create_dir(&dir.0).unwrap();
        dir
    }
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
impl Drop for CacheDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn backend() -> OnnxBackend {
    OnnxBackend::load(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/v0.onnx"),
        &cpu_config(),
    )
    .unwrap()
}

#[test]
fn batches_raw_outputs_and_attaches_only_requested_ownership() {
    let inputs: Vec<_> = (0..3)
        .map(|i| NNInput {
            board_dim: MAX_BOARD_DIM,
            spatial: [(i % 2) as u8; NUM_SPATIAL_FEATURES * MAX_BOARD_AREA],
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
        output.process_in_place(Player::White, &[true; MAX_POLICY_SIZE], MAX_BOARD_DIM);
        assert!((output.white_score_mean() - i as f32 * 20.0).abs() < 1e-6);
        if let Some(ownership) = output.white_ownership() {
            assert!((ownership[0] - ((i % 2) as f32).tanh()).abs() < 1e-6);
        }
    }
    // Reuse packing buffers with a different batch size and no ownership fetch.
    outputs.clear();
    backend.evaluate_batch(&&inputs[..1], &mut outputs).unwrap();
    assert_eq!(outputs.len(), 1);
    assert!(!outputs[0].has_ownership());
}

#[test]
fn smaller_boards_pack_inputs_expand_policy_and_keep_ownership_compact() {
    use crate::game::{game_state::GameState, rules::Rules};
    let mut backend = backend();
    let mut outputs = Vec::new();
    // Change shape repeatedly on the same loaded model and reusable buffers.
    for board_dim in [9, 3, 5, 1, 9] {
        let state = GameState::new(Rules {
            board_dim,
            ..Rules::default()
        });
        let mut input = NNInput::encode(&state);
        input.include_ownership = true;
        // Padding must never enter the model, even if its storage is nonzero.
        input.spatial.fill(u8::MAX);
        let mut expected_spatial = Vec::new();
        for plane in 0..NUM_SPATIAL_FEATURES {
            for y in 0..board_dim {
                for x in 0..board_dim {
                    let value = spatial_feature(plane * board_dim * board_dim + x + y * board_dim);
                    input.spatial[plane * MAX_BOARD_AREA + x + y * MAX_BOARD_DIM] = value;
                    expected_spatial.push(f32::from(value));
                }
            }
        }
        outputs.clear();
        backend.evaluate_batch(&[input], &mut outputs).unwrap();
        assert_eq!(
            backend.spatial.len(),
            NUM_SPATIAL_FEATURES * board_dim * board_dim
        );
        assert_eq!(backend.spatial, expected_spatial);
        let output = Arc::get_mut(&mut outputs[0]).unwrap();
        let legal = crate::inference::policy::legal_mask(&state);
        output.process_in_place(Player::White, &legal, board_dim);
        let policy = output.policy_probs();
        assert_eq!(policy.len(), board_dim * board_dim + 1);
        let ownership = output.white_ownership().unwrap();
        assert_eq!(ownership.len(), board_dim * board_dim);
        for y in 0..board_dim {
            for x in 0..board_dim {
                let i = x + y * board_dim;
                let logit = f32::from(spatial_feature(x + y * board_dim));
                if legal[i] {
                    assert!((policy[i] / policy[board_dim * board_dim] - logit.exp()).abs() < 1e-5);
                }
                assert!((ownership[x + y * board_dim] - logit.tanh()).abs() < 1e-6);
            }
        }
        assert!((policy.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}

#[cfg(debug_assertions)]
#[test]
fn debug_checks_reject_nonfinite_model_outputs() {
    let mut backend = backend();
    let mut outputs = Vec::new();
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        // Spatial bytes cannot contain NaN/Inf. The fixture feeds global floats
        // into value and pass logits, so invalid outputs still reach the check.
        let input = NNInput {
            board_dim: MAX_BOARD_DIM,
            spatial: [0; NUM_SPATIAL_FEATURES * MAX_BOARD_AREA],
            global: [invalid; NUM_GLOBAL_FEATURES],
            include_ownership: true,
        };
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.evaluate_batch(&[input], &mut outputs)
        }));
        assert!(panic.is_err());
        assert!(outputs.is_empty());
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
    assert!(OnnxBackend::load(&path, &cpu_config()).is_err());
}

#[test]
fn rejects_runtime_output_shapes_that_omit_pass() {
    use crate::game::{game_state::GameState, rules::Rules};
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/wrong_output.onnx");
    let mut backend = OnnxBackend::load(&path, &cpu_config()).unwrap();
    let input = NNInput::encode(&GameState::new(Rules {
        board_dim: 5,
        ..Rules::default()
    }));
    let mut outputs = Vec::new();
    let error = backend.evaluate_batch(&[input], &mut outputs).unwrap_err();
    let InferenceError::Onnx(error) = error else {
        panic!("expected shape error")
    };
    assert!(error.to_string().contains("invalid shape in policy_logits"));
    assert!(outputs.is_empty());
}

#[cfg(not(feature = "cuda"))]
#[test]
fn unavailable_cuda_returns_an_error_instead_of_falling_back_to_cpu() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/v0.onnx");
    assert!(
        OnnxBackend::load(
            &path,
            &ExecutorConfig {
                device: InferenceDevice::Cuda { device_id: 0 },
                ..cpu_config()
            }
        )
        .is_err()
    );
}

#[test]
fn invalid_executor_settings_return_errors_before_loading() {
    let path = std::path::Path::new("unused.onnx");
    for (config, message) in [
        (
            ExecutorConfig {
                intra_threads: 0,
                ..cpu_config()
            },
            "executor intra_threads must be positive",
        ),
        (
            ExecutorConfig {
                intra_threads: i32::MAX as usize + 1,
                ..cpu_config()
            },
            "executor intra_threads exceeds ONNX Runtime's i32 limit",
        ),
        (
            ExecutorConfig {
                device: InferenceDevice::Cuda { device_id: 0 },
                intra_threads: 0,
                ..cpu_config()
            },
            "executor intra_threads must be positive",
        ),
        (
            ExecutorConfig {
                device: InferenceDevice::Cuda { device_id: -1 },
                ..cpu_config()
            },
            "CUDA device_id must be nonnegative",
        ),
    ] {
        let error = OnnxBackend::load(path, &config)
            .err()
            .expect("invalid device must be rejected");
        assert!(error.to_string().contains(message));
    }
}

#[test]
fn unavailable_accelerators_are_rejected_before_loading() {
    let path = Path::new("unused.onnx");
    for (device, supported, message) in [
        (
            InferenceDevice::CoreMl {
                compute_units: CoreMlComputeUnits::All,
                model_cache_dir: None,
            },
            cfg!(all(
                feature = "coreml",
                any(target_os = "macos", target_os = "ios")
            )),
            "CoreML inference requires",
        ),
        (
            InferenceDevice::WebGpu {
                power_preference: WebGpuPowerPreference::HighPerformance,
                preferred_layout: WebGpuLayout::Nhwc,
            },
            cfg!(feature = "webgpu"),
            "WebGPU inference requires",
        ),
    ] {
        if supported {
            continue;
        }
        let error = OnnxBackend::load(
            path,
            &ExecutorConfig {
                device,
                ..cpu_config()
            },
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains(message), "{error}");
    }
}

#[cfg(any(all(feature = "coreml", target_os = "macos"), feature = "webgpu"))]
fn compare_accelerator_with_cpu(device: InferenceDevice) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/v0_conv.onnx");
    let mut accelerator = OnnxBackend::load(
        &path,
        &ExecutorConfig {
            device,
            ..cpu_config()
        },
    )
    .unwrap();
    let mut cpu = OnnxBackend::load(&path, &cpu_config()).unwrap();
    for (board_dim, batch_size, ownership) in
        [(9, 1, true), (13, 3, false), (19, 2, true), (9, 3, true)]
    {
        let inputs: Vec<_> = (0..batch_size)
            .map(|i| NNInput {
                board_dim,
                spatial: std::array::from_fn(|j| ((i + j / 7) % 2) as u8),
                global: [i as f32 * 0.25 - 0.75, 0.5],
                include_ownership: ownership && i % 2 == 0,
            })
            .collect();
        let mut expected = Vec::new();
        let mut actual = Vec::new();
        cpu.evaluate_batch(&inputs, &mut expected).unwrap();
        accelerator.evaluate_batch(&inputs, &mut actual).unwrap();
        assert_eq!(actual.len(), expected.len());
        let legal = vec![true; board_dim * board_dim + 1];
        for (expected, actual) in expected.iter_mut().zip(&mut actual) {
            let expected = Arc::get_mut(expected).unwrap();
            let actual = Arc::get_mut(actual).unwrap();
            expected.process_in_place(Player::White, &legal, board_dim);
            actual.process_in_place(Player::White, &legal, board_dim);
            let close = |a: f32, b: f32| {
                assert!(
                    (a - b).abs() <= 1e-4 * (1.0 + a.abs()),
                    "expected {a}, got {b}"
                )
            };
            for (&a, &b) in expected.policy_probs().iter().zip(actual.policy_probs()) {
                close(a, b);
            }
            close(expected.white_win_prob(), actual.white_win_prob());
            close(expected.white_score_mean(), actual.white_score_mean());
            close(expected.white_score_mean_sq(), actual.white_score_mean_sq());
            assert_eq!(expected.has_ownership(), actual.has_ownership());
            if let (Some(expected), Some(actual)) =
                (expected.white_ownership(), actual.white_ownership())
            {
                for (&a, &b) in expected.iter().zip(actual) {
                    close(a, b);
                }
            }
        }
    }
}

#[cfg(all(feature = "coreml", target_os = "macos"))]
#[test]
#[ignore = "requires macOS 12+ with CoreML; run explicitly on supported hardware"]
fn coreml_matches_cpu_across_dynamic_shapes() {
    let cache = CacheDir::new("coreml");
    let device = InferenceDevice::CoreMl {
        compute_units: CoreMlComputeUnits::All,
        model_cache_dir: Some(cache.0.clone()),
    };
    compare_accelerator_with_cpu(device.clone());
    assert!(
        std::fs::read_dir(&cache.0).unwrap().next().is_some(),
        "CoreML should cache at least one compiled partition"
    );
    // A new session must give the same outputs when loading cached partitions.
    compare_accelerator_with_cpu(device);
    for compute_units in [
        CoreMlComputeUnits::CpuAndGpu,
        CoreMlComputeUnits::CpuAndNeuralEngine,
        CoreMlComputeUnits::CpuOnly,
    ] {
        compare_accelerator_with_cpu(InferenceDevice::CoreMl {
            compute_units,
            model_cache_dir: None,
        });
    }
}

#[cfg(feature = "webgpu")]
#[test]
#[ignore = "requires a WebGPU-compatible GPU/driver; run explicitly on supported hardware"]
fn webgpu_matches_cpu_across_dynamic_shapes() {
    for power_preference in [
        WebGpuPowerPreference::HighPerformance,
        WebGpuPowerPreference::LowPower,
    ] {
        for preferred_layout in [WebGpuLayout::Nhwc, WebGpuLayout::Nchw] {
            compare_accelerator_with_cpu(InferenceDevice::WebGpu {
                power_preference,
                preferred_layout,
            });
        }
    }
}

#[test]
fn preserves_policy_order_including_pass() {
    let mut backend = backend();
    let input = NNInput {
        board_dim: MAX_BOARD_DIM,
        spatial: std::array::from_fn(spatial_feature),
        global: [0.0; NUM_GLOBAL_FEATURES],
        include_ownership: false,
    };
    let mut outputs = Vec::new();
    backend.evaluate_batch(&[input], &mut outputs).unwrap();
    let output = Arc::get_mut(&mut outputs[0]).unwrap();
    output.process_in_place(Player::White, &[true; MAX_POLICY_SIZE], MAX_BOARD_DIM);
    let probabilities = output.policy_probs();
    for i in 0..MAX_BOARD_AREA {
        assert!(
            (probabilities[i] / probabilities[0] - f32::from(spatial_feature(i)).exp()).abs()
                < 1e-5
        );
    }
    assert_eq!(
        probabilities[MAX_BOARD_DIM * MAX_BOARD_DIM],
        probabilities[0]
    );
}

fn spatial_feature(i: usize) -> u8 {
    ((i * 13 + i / 5 + i / 17) % 2) as u8
}
