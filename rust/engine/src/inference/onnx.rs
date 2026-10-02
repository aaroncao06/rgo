//! The V0 ONNX tensor boundary. No model architecture or search processing lives here.

use crate::game::board::MAX_BOARD_DIM;
use std::{path::Path, sync::Arc};

use ort::{
    ep,
    session::{HasSelectedOutputs, OutputSelector, RunOptions, Session, SessionOutputs},
    value::{Outlet, TensorElementType, TensorRef, ValueType},
};

use super::{
    backend::{InferenceBackend, InferenceError},
    inputs::{NNInput, NUM_GLOBAL_FEATURES, NUM_SPATIAL_FEATURES},
    outputs::NNOutput,
};

/// Select an execution provider, not exclusive ownership of a device.
#[derive(Debug, Clone, Copy, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum InferenceDevice {
    Cpu {
        intra_threads: usize,
    },
    /// Requires a CUDA-enabled ONNX Runtime and the crate's `cuda` feature.
    Cuda {
        device_id: i32,
    },
}

impl InferenceDevice {
    pub fn validate(self) -> Result<(), &'static str> {
        match self {
            Self::Cpu { intra_threads: 0 } => Err("CPU intra_threads must be positive"),
            Self::Cuda { device_id } if device_id < 0 => Err("CUDA device_id must be nonnegative"),
            Self::Cuda { .. } if !cfg!(feature = "cuda") => {
                Err("CUDA inference requires the cuda feature")
            }
            _ => Ok(()),
        }
    }
}

pub(crate) struct OnnxBackend {
    session: Session,
    without_ownership: RunOptions<HasSelectedOutputs>,
    // Each executor reuses its contiguous input buffers between batches.
    spatial: Vec<f32>,
    global: Vec<f32>,
}

impl OnnxBackend {
    /// Construct on the owning executor thread.
    /// CPU intra-op parallelism is explicit to avoid multiplying thread pools.
    pub(crate) fn load(path: &Path, device: InferenceDevice) -> ort::Result<Self> {
        device.validate().map_err(ort::Error::new)?;
        let (provider, intra_threads) = match device {
            InferenceDevice::Cpu { intra_threads } => (
                ep::CPU::default().with_arena_allocator(true).build(),
                intra_threads,
            ),
            #[cfg(feature = "cuda")]
            InferenceDevice::Cuda { device_id } => {
                (ep::CUDA::default().with_device_id(device_id).build(), 1)
            }
            #[cfg(not(feature = "cuda"))]
            InferenceDevice::Cuda { .. } => unreachable!("CUDA support was validated"),
        };
        let session = Session::builder()?
            .with_intra_threads(intra_threads)?
            // A requested provider must register successfully; don't silently
            // turn an unavailable CUDA configuration into a CPU-only session.
            .with_execution_providers([provider.error_on_failure()])?
            .commit_from_file(path)?;
        validate_contract(&session)?;
        let without_ownership = RunOptions::new()?.with_outputs(
            OutputSelector::no_default()
                .with("policy_logits")
                .with("value"),
        );
        Ok(Self {
            session,
            without_ownership,
            spatial: Vec::new(),
            global: Vec::new(),
        })
    }

    fn evaluate(
        &mut self,
        inputs: &[NNInput],
        outputs: &mut Vec<Arc<NNOutput>>,
    ) -> ort::Result<()> {
        debug_assert!(outputs.is_empty());
        debug_assert!(!inputs.is_empty(), "cannot evaluate an empty batch");
        let board_dim = inputs[0].board_dim;
        debug_assert!((1..=MAX_BOARD_DIM).contains(&board_dim));
        debug_assert!(
            inputs.iter().all(|input| input.board_dim == board_dim),
            "batch mixes board sizes"
        );
        let board_area = board_dim * board_dim;
        let policy_size = board_area + 1;
        self.spatial.clear();
        self.global.clear();
        self.spatial
            .reserve(inputs.len() * NUM_SPATIAL_FEATURES * board_area);
        self.global.reserve(inputs.len() * NUM_GLOBAL_FEATURES);
        for input in inputs {
            for row in input.spatial_rows() {
                self.spatial.extend_from_slice(row);
            }
            self.global.extend_from_slice(&input.global);
        }
        let n = inputs.len();
        let include_ownership = inputs.iter().any(|input| input.include_ownership);
        let tensors = ort::inputs![
            "spatial" => TensorRef::from_array_view(([n, NUM_SPATIAL_FEATURES, board_dim, board_dim], self.spatial.as_slice()))?,
            "global" => TensorRef::from_array_view(([n, NUM_GLOBAL_FEATURES], self.global.as_slice()))?,
        ];
        let batch = if include_ownership {
            self.session.run(tensors)?
        } else {
            self.session
                .run_with_options(tensors, &self.without_ownership)?
        };
        // Validate dynamic output shapes before copying each compact result.
        let policy_batch = tensor(&batch, "policy_logits", &[n, policy_size])?;
        let value_batch = tensor(&batch, "value", &[n, 3])?;
        let ownership_batch = if include_ownership {
            Some(tensor(
                &batch,
                "ownership_logits",
                &[n, 1, board_dim, board_dim],
            )?)
        } else {
            None
        };
        for (i, input) in inputs.iter().enumerate() {
            let compact_policy = &policy_batch[i * policy_size..(i + 1) * policy_size];
            let value_row = &value_batch[i * 3..(i + 1) * 3];
            let mut output = NNOutput::from_raw(
                compact_policy.into(),
                value_row[0],
                value_row[1],
                value_row[2],
            );
            if input.include_ownership {
                let ownership_logits =
                    &ownership_batch.unwrap()[i * board_area..(i + 1) * board_area];
                // Allocate the ownership Arc directly from the tensor slice.
                output = output.with_ownership_logits(ownership_logits.into());
            }
            outputs.push(Arc::new(output));
        }
        Ok(())
    }
}

impl InferenceBackend for OnnxBackend {
    fn evaluate_batch(
        &mut self,
        inputs: &[NNInput],
        outputs: &mut Vec<Arc<NNOutput>>,
    ) -> Result<(), InferenceError> {
        self.evaluate(inputs, outputs)
            .map_err(|error| InferenceError::Onnx(Arc::new(error)))
    }
}

fn validate_contract(session: &Session) -> ort::Result<()> {
    if session.metadata()?.custom("rgo.io_version").as_deref() != Some("0") {
        return Err(ort::Error::new("expected model metadata rgo.io_version=0"));
    }
    validate_ports(
        session.inputs(),
        &[
            ("spatial", &[NUM_SPATIAL_FEATURES as i64, -1, -1]),
            ("global", &[NUM_GLOBAL_FEATURES as i64]),
        ],
    )?;
    validate_ports(
        session.outputs(),
        &[
            ("policy_logits", &[-1]),
            ("value", &[3]),
            ("ownership_logits", &[1, -1, -1]),
        ],
    )
}

fn validate_ports(ports: &[Outlet], expected: &[(&str, &[i64])]) -> ort::Result<()> {
    if ports.len() != expected.len() {
        return Err(ort::Error::new(
            "unexpected number of model inputs or outputs",
        ));
    }
    for &(name, dimensions) in expected {
        let port = ports
            .iter()
            .find(|port| port.name() == name)
            .ok_or_else(|| ort::Error::new(format!("missing tensor {name}")))?;
        let ValueType::Tensor {
            ty: TensorElementType::Float32,
            shape,
            ..
        } = port.dtype()
        else {
            return Err(ort::Error::new(format!("{name} must be a float32 tensor")));
        };
        if shape.len() != dimensions.len() + 1
            || shape[0] != -1
            || shape[1..]
                .iter()
                .zip(dimensions)
                .any(|(&actual, &expected)| actual != expected)
        {
            return Err(ort::Error::new(format!(
                "invalid shape for {name}: {shape:?}"
            )));
        }
    }
    Ok(())
}

fn tensor<'a>(
    outputs: &'a SessionOutputs<'_>,
    name: &str,
    expected: &[usize],
) -> ort::Result<&'a [f32]> {
    let output = outputs
        .get(name)
        .ok_or_else(|| ort::Error::new(format!("missing output {name}")))?;
    let (shape, values) = output.try_extract_tensor::<f32>()?;
    // Runtime shapes can disagree with declarations even after load validation.
    if shape.len() != expected.len()
        || shape
            .iter()
            .zip(expected)
            .any(|(&actual, &expected)| actual != expected as i64)
    {
        return Err(ort::Error::new(format!(
            "invalid shape in {name}: expected {expected:?}, got {shape:?}"
        )));
    }
    debug_assert!(
        values.iter().all(|value| value.is_finite()),
        "nonfinite values in {name}"
    );
    Ok(values)
}

#[cfg(test)]
mod tests;
