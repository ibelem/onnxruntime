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