# Security finding #766: `output_name = output_name.substr(0, output_name.find('/'))` (line …

**Summary:** `output_name = output_name.substr(0, output_name.find('/'))` (line …

**CWE IDs:** CWE-697: Incorrect Comparison (name truncation) leading to CWE-787 wrong-slot / OOB write
**Severity / Impact:** The constant-folded node's bytes are written into the caller-owned ORT output buffer at the wrong positional index. Combined with the missing size check in FillOutputHelper (see companion finding), this silently corrupts an adjacent output tensor and, when the wrong slot's declared dtype is narrower than the constant node's dtype, overflows that slot. Reachable from a compromised renderer whenever the OV EP runs a partially-constant subgraph (device != NPU, !is_wholly_supported_graph).
**Affected location:** `targets/onnxruntime/onnxruntime/core/providers/openvino/backend_utils.cc:99` — `GetOutputTensor()`
**Validated for repos:** onnxruntime
**Trust boundary:** WebNN Mojo IPC → GPU/utility process; the renderer-supplied graph's output node names survive ConstantFolding into const_outputs_map_ and are matched against ONNX output names here.

## Description / Root cause
`output_name = output_name.substr(0, output_name.find('/'))` (line 100-102) truncates the OV result's friendly name at the first '/', then looks the truncated string up in `output_names`. The lookup is an exact-string match on a lossily-truncated key: an OV result friendly name like `Foo/Identity:0` collapses to `Foo`, and any *different* ONNX output literally named `Foo` will match, so `it->second` returns the ONNX positional index of the wrong output. There is no verification that the truncated prefix uniquely and correctly identifies the intended ONNX output; the only guard (line 105-106) fires when NO match exists, not when a WRONG match exists.

**Validator analysis:** The truncation-then-exact-match in GetOutputTensor (backend_utils.cc:100-108) is real and unconditional whenever const_outputs_map_ is non-empty (partially-constant subgraph, device!=NPU, !is_wholly_supported_graph). ONNX output names are attacker-controlled: GetOperandName (graph_builder_ort.cc:172-177) appends "_<id>" but SanitizeName does not strip '/', so a compromised renderer can name a constant output "bar_1/x" (id 5 -> "bar_1/x_5") and a second output "bar" that gets id 1 -> "bar_1"; truncation of the former to "bar_1" resolves to the latter's positional index. The only guard (line 105-106) fires on NO match, never on a WRONG match. GetOutput is then called with index=WRONG-slot but shape=node->get_shape() (the constant's shape, line 109), and FillOutputHelper (line 204-208) std::copy's the constant's element count into that slot's buffer with no bounds check. Because WebNN dispatch pre-binds fixed-size output tensors (MLTensor) sized to each output's declared shape, writing the larger constant into the smaller wrong slot is an out-of-bounds heap write -> CWE-697 leading to CWE-787. vulnType/impact are accurate. The 'unverified hop' (does OV emit '/'-bearing friendly names) is resolved because the '/' originates from the attacker's operand name, which OV preserves in the Result friendly name; the truncation code exists precisely to strip such suffixes. The proposed fix is correct in direction: matching the full OV friendly name, or storing the ONNX output index directly on the const node in CreateOVModel, eliminates the lossy reconstruction; additionally the dtype/shape-equality check before GetOutput is a sound belt-and-suspenders, and the companion size check in FillOutputHelper should also be added so a size mismatch throws rather than overflows. The observable effect is a heap OOB write (crash/ASan), which is not cleanly assertable as a spec throw, so no WPT test; and the exact OperandId ordering + constant-folding conditions make a guaranteed self-contained browser trigger unreliable, so reproduction is given as steps.

## Exploit / Proof of Concept
A renderer crafts a WebNN graph (mapped to ORT→OV EP) with two outputs where one output node's name, after OV assigns a friendly name of the form `<name>/<suffix>` during ConstantFolding, truncates to a prefix that exactly equals a second output's ONNX name. During Infer (basic_backend.cc:410-414) GetOutputTensor returns the index of the second output; FillOutputsWithConstantData then writes the first (constant) node's data into the second output's buffer. Whether OV emits '/'-bearing friendly names for these result nodes was not byte-verified here (unverified hop), but the truncation-then-exact-match logic is unconditionally applied.

## Reproduction (steps)
```
Not cleanly web-reachable as a single guaranteed HTML page because it requires (a) a constant-foldable output and (b) an exact ONNX-name collision that depends on the internal OperandId assignment order. Steps (from a compromised renderer / MLGraphBuilder):
1. Build an MLContext/MLGraphBuilder targeting a non-NPU device (CPU/GPU) so the OV EP runs a partially-supported graph (is_wholly_supported_graph == false).
2. Create the FIRST operand so it receives OperandId 1 and expose it as a graph output named "bar" -> ONNX output name becomes "bar_1". Give it a SMALL shape (e.g. [1]).
3. Create a second output that constant-folds to a pure ov::op::v0::Constant (e.g. builder.relu(constantOperand) or a cast of a constant) with a LARGE shape (e.g. [1024]) and name it "bar_1/x" -> ONNX output name "bar_1/x_<id>".
4. build({ "bar": out1, "bar_1/x": constOut }) and dispatch with pre-allocated MLTensor outputs sized to each declared shape.
5. During Infer, CreateOVModel stores the constant Result under friendly name "bar_1/x..." in const_outputs_map_; GetOutputTensor truncates it to "bar_1", matches output "bar_1" (slot of out1, buffer sized [1]), and FillOutputsWithConstantData copies the 1024-element constant into the 1-element buffer -> heap OOB write (observable under ASan as heap-buffer-overflow in FillOutputHelper / std::copy).
```

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-697 -> CWE-787 in
// onnxruntime/core/providers/openvino/backend_utils.cc:100-108 (GetOutputTensor)
// and the unbounded copy in FillOutputHelper (backend_utils.cc:204-208).
//
// Pre-fix: GetOutputTensor truncates the OV Result friendly name at the first
// '/' and does an exact-string lookup in output_names, so a constant output
// whose ONNX name is "bar_1/x_5" resolves to the positional index of a
// DIFFERENT output literally named "bar_1"; FillOutputHelper then std::copy's
// the constant's (larger) data into that wrong, smaller, pre-bound buffer
// (ASan heap-buffer-overflow). Post-fix: the full name must be matched (no
// truncation) OR a shape/dtype mismatch must ORT_THROW before GetOutput.
//
// NOTE: core/providers/openvino has no standalone gtest harness in this tree,
// and GetOutputTensor takes an Ort::KernelContext + SubGraphContext map that
// require a live OV-EP session, so this is a SKELETON. TODO markers name the
// missing pieces.

#include "gtest/gtest.h"

// TODO: include the real headers once a link target exists, e.g.
// #include "core/providers/openvino/backend_utils.h"
// #include "core/providers/openvino/contexts.h"

namespace onnxruntime {
namespace openvino_ep {
namespace test {

TEST(OpenVINOBackendUtils, GetOutputTensor_TruncatedNameMustNotMatchWrongSlot) {
  // TODO: build a SubGraphContext::string_index_map_t with a name collision:
  //   output_names["bar_1"]     = 0;  // small declared output
  //   output_names["bar_1/x_5"] = 1;  // constant-folded output (large)
  //
  // TODO: construct an ov::op::v0::Constant-backed node whose shape is LARGER
  //   than slot 0's declared shape and whose friendly name is "bar_1/x_5".
  //
  // TODO: build a fake Ort::KernelContext whose GetOutput returns a fixed-size
  //   buffer for each index (mirrors WebNN pre-bound MLTensor outputs).
  //
  // Expectation encoding the fix: GetOutputTensor(context, "bar_1/x_5",
  //   output_names, constNode) must EITHER resolve to index 1 (full-name match)
  //   OR ORT_THROW on the shape mismatch — it must NOT silently return index 0.
  //
  // EXPECT_THROW(
  //   GetOutputTensor(ctx, "bar_1/x_5", output_names, constNode),
  //   onnxruntime::OnnxRuntimeException);
  //
  // Pre-fix this returns index 0 and the subsequent FillOutputsWithConstantData
  // overflows slot 0's buffer (ASan). Post-fix it either targets index 1 or
  // throws.
  GTEST_SKIP() << "Skeleton: requires OV-EP session fixtures (see TODOs).";
}

}  // namespace test
}  // namespace openvino_ep
}  // namespace onnxruntime
```
**Build / run:** core/providers/openvino has no dedicated gtest binary in this tree; wire the file into the onnxruntime provider test target (e.g. onnxruntime_test_all) and run: onnxruntime_test_all --gtest_filter=OpenVINOBackendUtils.GetOutputTensor_TruncatedNameMustNotMatchWrongSlot. With ASan, the pre-fix build reports 'heap-buffer-overflow WRITE' inside FillOutputHelper<T> / std::copy (backend_utils.cc:208); post-fix the call resolves to the correct slot or throws OnnxRuntimeException and the test passes. Replace all TODO fixtures before running.

## Suggested fix
Do not truncate; match the full OV friendly name against a map that stores the exact OV result names, or store the ONNX output index directly on the const node when populating const_outputs_map_ in CreateOVModel (so no name reconstruction is needed at Infer time). If truncation is retained, verify the resolved ONNX output's declared shape and dtype equal the node's shape/dtype before calling GetOutput, and ORT_THROW on mismatch.

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `claude-4-6-sonnet` |
| Researcher | `claude-4-8-opus` |
| Validator | `claude-4-8-opus` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #766.
