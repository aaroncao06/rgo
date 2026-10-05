//! The V0 ONNX tensor boundary. No model architecture or search processing lives here.

use crate::game::board::MAX_BOARD_DIM;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use ort::{
    ep,
    session::{HasSelectedOutputs, OutputSelector, RunOptions, Session, SessionOutputs},
    value::{Outlet, TensorElementType, TensorRef, ValueType},
};

use super::{
    backend::{InferenceBackend, InferenceError, InputBatch},
    inputs::{NUM_GLOBAL_FEATURES, NUM_SPATIAL_FEATURES},
    outputs::NNOutput,
    runtime::ExecutorConfig,
};

/// Select an execution provider, not exclusive ownership of a device.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum InferenceDevice {
    Cpu {},
    /// Requires a CUDA-enabled ONNX Runtime and the crate's `cuda` feature.
    // TensorRT is deferred; benchmark it against CUDA when NVIDIA hardware is available.
    Cuda {
        device_id: i32,
    },
    /// Requires the `coreml` feature and macOS 12+ (MLProgram format).
    #[serde(rename = "coreml")]
    CoreMl {
        #[serde(default)]
        compute_units: CoreMlComputeUnits,
        /// Cache of converted/compiled models, separate from evaluation results.
        #[serde(default)]
        model_cache_dir: Option<PathBuf>,
    },
    /// Requires the `webgpu` feature and a compatible GPU/driver.
    #[serde(rename = "webgpu")]
    WebGpu {
        #[serde(default)]
        power_preference: WebGpuPowerPreference,
        #[serde(default)]
        preferred_layout: WebGpuLayout,
    },
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoreMlComputeUnits {
    #[default]
    All,
    CpuAndGpu,
    CpuAndNeuralEngine,
    CpuOnly,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WebGpuPowerPreference {
    #[default]
    HighPerformance,
    LowPower,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WebGpuLayout {
    Nchw,
    #[default]
    Nhwc,
}

impl InferenceDevice {
    pub fn validate(&self) -> Result<(), &'static str> {
        match self {
            Self::Cuda { device_id } if *device_id < 0 => Err("CUDA device_id must be nonnegative"),
            Self::Cuda { .. } if !cfg!(feature = "cuda") => {
                Err("CUDA inference requires the cuda feature")
            }
            Self::CoreMl {
                model_cache_dir: Some(path),
                ..
            } if path.as_os_str().is_empty() => Err("CoreML model_cache_dir must not be empty"),
            Self::CoreMl {
                model_cache_dir: Some(path),
                ..
            } if path.to_str().is_none() => Err("CoreML model_cache_dir must be valid UTF-8"),
            Self::CoreMl { .. } if !cfg!(feature = "coreml") => {
                Err("CoreML inference requires the coreml feature")
            }
            Self::CoreMl { .. } if !cfg!(any(target_os = "macos", target_os = "ios")) => {
                Err("CoreML inference requires an Apple platform")
            }
            Self::WebGpu { .. } if !cfg!(feature = "webgpu") => {
                Err("WebGPU inference requires the webgpu feature")
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
    ownership_requests: Vec<bool>,
}

impl OnnxBackend {
    /// Construct on the owning executor thread.
    /// CPU intra-op parallelism is explicit to avoid multiplying thread pools.
    pub(crate) fn load(path: &Path, config: &ExecutorConfig) -> ort::Result<Self> {
        config.validate().map_err(ort::Error::new)?;
        let provider = match &config.device {
            InferenceDevice::Cpu {} => ep::CPU::default().with_arena_allocator(true).build(),
            #[cfg(feature = "cuda")]
            InferenceDevice::Cuda { device_id } => {
                ep::CUDA::default().with_device_id(*device_id).build()
            }
            #[cfg(not(feature = "cuda"))]
            InferenceDevice::Cuda { .. } => unreachable!("CUDA support was validated"),
            #[cfg(feature = "coreml")]
            InferenceDevice::CoreMl {
                compute_units,
                model_cache_dir,
            } => {
                let units = match compute_units {
                    CoreMlComputeUnits::All => ep::coreml::ComputeUnits::All,
                    CoreMlComputeUnits::CpuAndGpu => ep::coreml::ComputeUnits::CPUAndGPU,
                    CoreMlComputeUnits::CpuAndNeuralEngine => {
                        ep::coreml::ComputeUnits::CPUAndNeuralEngine
                    }
                    CoreMlComputeUnits::CpuOnly => ep::coreml::ComputeUnits::CPUOnly,
                };
                let mut provider = ep::CoreML::default()
                    .with_model_format(ep::coreml::ModelFormat::MLProgram)
                    .with_compute_units(units)
                    .with_static_input_shapes(false);
                if let Some(dir) = model_cache_dir {
                    provider = provider
                        .with_model_cache_dir(dir.to_str().expect("cache path was validated"));
                }
                provider.build()
            }
            #[cfg(not(feature = "coreml"))]
            InferenceDevice::CoreMl { .. } => unreachable!("CoreML support was validated"),
            #[cfg(feature = "webgpu")]
            InferenceDevice::WebGpu {
                power_preference,
                preferred_layout,
            } => {
                use ep::ArbitrarilyConfigurableExecutionProvider;
                let power = match power_preference {
                    WebGpuPowerPreference::HighPerformance => "high-performance",
                    WebGpuPowerPreference::LowPower => "low-power",
                };
                let layout = match preferred_layout {
                    WebGpuLayout::Nchw => ep::webgpu::PreferredLayout::NCHW,
                    WebGpuLayout::Nhwc => ep::webgpu::PreferredLayout::NHWC,
                };
                ep::WebGPU::default()
                    .with_arbitrary_config("ep.webgpuexecutionprovider.powerPreference", power)
                    .with_preferred_layout(layout)
                    .with_enable_graph_capture(false)
                    .build()
            }
            #[cfg(not(feature = "webgpu"))]
            InferenceDevice::WebGpu { .. } => unreachable!("WebGPU support was validated"),
        };
        let session = Session::builder()?
            .with_intra_threads(config.intra_threads)?
            // A requested provider must register successfully; don't silently
            // turn an unavailable provider into a CPU-only session.
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
            ownership_requests: Vec::new(),
        })
    }

    fn evaluate(
        &mut self,
        inputs: &dyn InputBatch,
        outputs: &mut Vec<Arc<NNOutput>>,
    ) -> ort::Result<()> {
        debug_assert!(outputs.is_empty());
        debug_assert!(!inputs.is_empty(), "cannot evaluate an empty batch");
        let board_dim = inputs.board_dim();
        debug_assert!((1..=MAX_BOARD_DIM).contains(&board_dim));
        let board_area = board_dim * board_dim;
        let policy_size = board_area + 1;
        self.spatial.clear();
        self.global.clear();
        self.ownership_requests.clear();
        self.spatial
            .reserve(inputs.len() * NUM_SPATIAL_FEATURES * board_area);
        self.global.reserve(inputs.len() * NUM_GLOBAL_FEATURES);
        self.ownership_requests.reserve(inputs.len());
        inputs.for_each_input(&mut |input| {
            debug_assert_eq!(input.board_dim, board_dim, "batch mixes board sizes");
            for row in input.spatial_rows() {
                self.spatial.extend(row.iter().copied().map(f32::from));
            }
            self.global.extend_from_slice(&input.global);
            self.ownership_requests.push(input.include_ownership);
        });
        let n = inputs.len();
        let include_ownership = self.ownership_requests.iter().any(|&include| include);
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
        for (i, &include_ownership) in self.ownership_requests.iter().enumerate() {
            let compact_policy = &policy_batch[i * policy_size..(i + 1) * policy_size];
            let value_row = &value_batch[i * 3..(i + 1) * 3];
            let mut output = NNOutput::from_raw(
                compact_policy.into(),
                value_row[0],
                value_row[1],
                value_row[2],
            );
            if include_ownership {
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
        inputs: &dyn InputBatch,
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
