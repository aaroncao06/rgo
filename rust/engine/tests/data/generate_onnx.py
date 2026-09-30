"""Regenerate the tiny deterministic ONNX fixtures (requires Python onnx).

This is a tensor-adapter fixture, not a proposed network architecture.
"""
from pathlib import Path

import onnx
from onnx import TensorProto, helper


def tensor(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


nodes = [
    helper.make_node("ReduceMean", ["global"], ["mean"], axes=[1], keepdims=1),
    helper.make_node("Concat", ["mean", "mean", "mean"], ["value"], axis=1),
    helper.make_node("Flatten", ["spatial"], ["flat"], axis=1),
    helper.make_node("Slice", ["flat", "start", "end", "axis"], ["policy_logits"]),
    helper.make_node("Slice", ["flat", "start", "ownership_end", "axis"], ["ownership_flat"]),
    helper.make_node("Reshape", ["ownership_flat", "ownership_shape"], ["ownership_logits"]),
]
initializers = [
    helper.make_tensor("start", TensorProto.INT64, [1], [0]),
    helper.make_tensor("end", TensorProto.INT64, [1], [82]),
    helper.make_tensor("axis", TensorProto.INT64, [1], [1]),
    helper.make_tensor("ownership_end", TensorProto.INT64, [1], [81]),
    helper.make_tensor("ownership_shape", TensorProto.INT64, [4], [-1, 1, 9, 9]),
]
graph = helper.make_graph(
    nodes, "rgo_v0_fixture",
    [tensor("spatial", ["N", 3, 9, 9]), tensor("global", ["N", 2])],
    [tensor("policy_logits", ["N", 82]), tensor("value", ["N", 3]),
     tensor("ownership_logits", ["N", 1, 9, 9])],
    initializer=initializers,
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
directory = Path(__file__).parent
for filename, version in [("v0.onnx", "0"), ("wrong_version.onnx", "1")]:
    helper.set_model_props(model, {"rgo.io_version": version})
    onnx.checker.check_model(model)
    onnx.save(model, directory / filename)

helper.set_model_props(model, {"rgo.io_version": "0"})
model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 4
onnx.checker.check_model(model)
onnx.save(model, directory / "wrong_shape.onnx")
