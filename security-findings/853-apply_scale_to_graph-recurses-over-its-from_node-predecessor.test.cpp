// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-674 in
// onnxruntime/core/providers/openvino/qdq_transformations/qdq_scales_fix.cc:87-125
// (GraphNode::apply_scale_to_graph). Encodes the fix: a very deep, acyclic
// DequantizeLinear -> Add_1 -> ... -> Add_N -> QuantizeLinear -> DequantizeLinear
// QDQ chain must NOT crash the process via unbounded recursion. Pre-fix this
// overflows the stack (SIGSEGV / ASan stack-overflow); post-fix the depth cap
// throws a clean onnxruntime error (ORT_THROW) that surfaces as an ov/ORT
// exception instead of a crash.
//
// NOTE: qdq_scales_fix::Transform takes a GraphViewer over an onnxruntime::Model.
// This skeleton builds an ONNX ModelProto with a long linear Add chain wrapped in
// a QDQ pair whose Q-scale/DQ-scale ratio exceeds the internal threshold so that
// scale_graph() reaches apply_scale_to_graph() at qdq_scales_fix.cc:737.
#include "gtest/gtest.h"
#include "core/graph/model.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/openvino/qdq_transformations/qdq_scales_fix.h"

using namespace onnxruntime;

namespace {
// TODO(reviewer): confirm helper for building an in-memory Model with initializers
// in this test tree (grep onnxruntime/test/providers/openvino or
// onnxruntime/test/framework for a Model/GraphProto builder helper such as
// CreateModel / ModelTestBuilder). Build:
//   DQ(scale=1e-3) -> Add_1 -> Add_2 -> ... -> Add_N -> Q(scale=1.0) -> DQ_out
// with N large enough to exhaust the stack pre-fix (e.g. 200000) and each Add
// consuming the previous node's single output plus a constant initializer.
std::unique_ptr<Model> BuildDeepQdqChain(int N /*chain length*/) {
  // TODO: assemble GraphProto nodes + float scale/zero-point initializers so that
  //  - the Q node's to_node[0] is a DequantizeLinear (qdq_scales_fix.cc:724-725)
  //  - q_scale / q_scale_factor > threshold(1.0) (qdq_scales_fix.cc:731)
  //  - device path is the GPU/uint16 branch selection is irrelevant here since we
  //    call Transform() directly.
  (void)N;
  return nullptr;
}
}  // namespace

TEST(OvepQdqScalesFix, DeepAddChainDoesNotStackOverflow) {
  auto model = BuildDeepQdqChain(/*N=*/200000);
  ASSERT_NE(model, nullptr);
  GraphViewer viewer(model->MainGraph());
  std::unique_ptr<onnxruntime::Model> out;
  const auto& logger = logging::LoggingManager::DefaultLogger();
  // Pre-fix: unbounded recursion in apply_scale_to_graph -> stack overflow (crash).
  // Post-fix: depth cap throws, Transform returns a non-OK Status or throws an
  // ORT exception — either way the process stays alive and the test observes it.
  EXPECT_ANY_THROW({
    Status s = qdq_scales_fix::Transform(viewer, logger, out);
    // If the fix returns a Status instead of throwing, assert it is not OK:
    ASSERT_FALSE(s.IsOK());
  });
}
