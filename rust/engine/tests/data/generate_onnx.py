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
    helper.make_node("Slice", ["spatial", "start", "channel_end", "axis"], ["ownership_logits"]),
    helper.make_node("Flatten", ["ownership_logits"], ["board_policy"], axis=1),
    helper.make_node("Mul", ["mean", "zero"], ["pass_logit"]),
    helper.make_node("Concat", ["board_policy", "pass_logit"], ["policy_logits"], axis=1),
]
initializers = [
    helper.make_tensor("start", TensorProto.INT64, [1], [0]),
    helper.make_tensor("channel_end", TensorProto.INT64, [1], [1]),
    helper.make_tensor("axis", TensorProto.INT64, [1], [1]),
    helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0]),
]
graph = helper.make_graph(
    nodes, "rgo_v0_fixture",
    [tensor("spatial", ["N", 3, "H", "W"]), tensor("global", ["N", 2])],
    [tensor("policy_logits", ["N", "P"]), tensor("value", ["N", 3]),
     tensor("ownership_logits", ["N", 1, "H", "W"])],
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

# Declared dynamic dimensions cannot prove that the graph returns H*W+1.
# This graph deliberately omits pass so runtime shape checking must reject it.
model.graph.input[0].type.tensor_type.shape.dim[1].dim_value = 3
model.graph.node[-1].CopyFrom(helper.make_node("Identity", ["board_policy"], ["policy_logits"]))
onnx.checker.check_model(model)
onnx.save(model, directory / "wrong_output.onnx")

# An accelerator smoke fixture with a constant-weight 1x1 convolution. The
# original adapter fixture has no convolution and is not a throughput benchmark.
conv_nodes = list(nodes)
conv_nodes[2] = helper.make_node(
    "Conv", ["spatial", "conv_weights"], ["ownership_logits"], kernel_shape=[1, 1],
    pads=[0, 0, 0, 0], strides=[1, 1], dilations=[1, 1],
)
conv_graph = helper.make_graph(
    conv_nodes, "rgo_v0_conv_fixture",
    [tensor("spatial", ["N", 3, "H", "W"]), tensor("global", ["N", 2])],
    [tensor("policy_logits", ["N", "P"]), tensor("value", ["N", 3]),
     tensor("ownership_logits", ["N", 1, "H", "W"])],
    initializer=[initializers[-1], helper.make_tensor(
        "conv_weights", TensorProto.FLOAT, [1, 3, 1, 1], [0.125, 0.25, 0.5]
    )],
)
conv_model = helper.make_model(
    conv_graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10
)
helper.set_model_props(conv_model, {"rgo.io_version": "0"})
onnx.checker.check_model(conv_model)
onnx.save(conv_model, directory / "v0_conv.onnx")
