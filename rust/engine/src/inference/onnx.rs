//! The V0 ONNX tensor boundary. No model architecture or search processing lives here.

use std::{path::Path, sync::Arc};

use ort::{
    session::{HasSelectedOutputs, OutputSelector, RunOptions, Session, SessionOutputs},
    value::{Outlet, TensorElementType, TensorRef, ValueType},
};

use super::{
    backend::{InferenceBackend, InferenceError},
    inputs::{NNInput, NUM_GLOBAL_FEATURES, NUM_SPATIAL_FEATURES},
    outputs::NNOutput,
    policy::{BOARD_POLICY_SIZE, POLICY_SIZE},
};
use crate::game::board::BOARD_SIZE;

pub(crate) struct OnnxBackend {
    session: Session,
    without_ownership: RunOptions<HasSelectedOutputs>,
    // Each executor reuses its contiguous input buffers between batches.
    spatial: Vec<f32>,
    global: Vec<f32>,
}

impl OnnxBackend {
    /// Construct on the executor thread, just like other backend factories.
    /// CPU intra-op parallelism is explicit to avoid multiplying thread pools.
    pub(crate) fn load(path: &Path, intra_threads: usize) -> ort::Result<Self> {
        assert!(intra_threads > 0, "CPU thread count must be positive");
        let session = Session::builder()?
            .with_intra_threads(intra_threads)?
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
        self.spatial.clear();
        self.global.clear();
        self.spatial
            .reserve(inputs.len() * NUM_SPATIAL_FEATURES * BOARD_POLICY_SIZE);
        self.global.reserve(inputs.len() * NUM_GLOBAL_FEATURES);
        for input in inputs {
            self.spatial.extend_from_slice(&input.spatial);
            self.global.extend_from_slice(&input.global);
        }
        let n = inputs.len();
        let include_ownership = inputs.iter().any(|input| input.include_ownership);
        let tensors = ort::inputs![
            "spatial" => TensorRef::from_array_view(([n, NUM_SPATIAL_FEATURES, BOARD_SIZE, BOARD_SIZE], self.spatial.as_slice()))?,
            "global" => TensorRef::from_array_view(([n, NUM_GLOBAL_FEATURES], self.global.as_slice()))?,
        ];
        let batch = if include_ownership {
            self.session.run(tensors)?
        } else {
            self.session
                .run_with_options(tensors, &self.without_ownership)?
        };
        // Extract raw tensors; debug checks enforce the exporter/backend contract.
        let policy_batch = tensor(&batch, "policy_logits", &[n, POLICY_SIZE])?;
        let value_batch = tensor(&batch, "value", &[n, 3])?;
        let ownership_batch = if include_ownership {
            Some(tensor(
                &batch,
                "ownership_logits",
                &[n, 1, BOARD_SIZE, BOARD_SIZE],
            )?)
        } else {
            None
        };
        for (i, input) in inputs.iter().enumerate() {
            let policy_logits = policy_batch[i * POLICY_SIZE..(i + 1) * POLICY_SIZE]
                .try_into()
                .unwrap();
            let value_row = &value_batch[i * 3..(i + 1) * 3];
            let mut output =
                NNOutput::from_raw(policy_logits, value_row[0], value_row[1], value_row[2]);
            if input.include_ownership {
                let ownership_logits = ownership_batch.unwrap()
                    [i * BOARD_POLICY_SIZE..(i + 1) * BOARD_POLICY_SIZE]
                    .try_into()
                    .unwrap();
                output = output.with_ownership_logits(ownership_logits);
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
            ("spatial", &[NUM_SPATIAL_FEATURES, BOARD_SIZE, BOARD_SIZE]),
            ("global", &[NUM_GLOBAL_FEATURES]),
        ],
    )?;
    validate_ports(
        session.outputs(),
        &[
            ("policy_logits", &[POLICY_SIZE]),
            ("value", &[3]),
            ("ownership_logits", &[1, BOARD_SIZE, BOARD_SIZE]),
        ],
    )
}

fn validate_ports(ports: &[Outlet], expected: &[(&str, &[usize])]) -> ort::Result<()> {
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
                .any(|(&actual, &expected)| actual != expected as i64)
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
    debug_assert!(
        shape.len() == expected.len()
            && shape
                .iter()
                .zip(expected)
                .all(|(&actual, &expected)| actual == expected as i64),
        "invalid shape in {name}: expected {expected:?}, got {shape:?}"
    );
    debug_assert!(
        values.iter().all(|value| value.is_finite()),
        "nonfinite values in {name}"
    );
    Ok(values)
}

#[cfg(test)]
mod tests;
