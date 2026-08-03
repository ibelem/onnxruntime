// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-125 OOB read in
//   onnxruntime/core/providers/openvino/qdq_transformations/qdq_scales_fix.cc:186
//   GraphNode::up_propagate_scale(), Mul/MatMul !find_dq branch:
//       from_node.back()->from_node[0]->up_propagate_scale();
// Pre-fix: a Mul/MatMul with no DequantizeLinear parent whose relevant
// parent is a graph input (empty from_node) OR whose inputs are all constant
// initializers (empty from_node) causes .back() on an empty vector and/or
// operator[] on an empty vector -> OOB read + wild-pointer deref (ASan:
// container-overflow / SEGV). Post-fix (non-empty guards + terminal handling
// of graph-input/initializer operands) the transform completes and returns OK.
//
// Entry point under test is the public header API:
//   onnxruntime::openvino_ep::qdq_scales_fix::Transform(GraphViewer, logger, model)
// which runs scale_graph unconditionally (device gating happens in the caller
// backend_manager.cc:479-503, not inside Transform), so no GPU is required to
// drive the vulnerable code in-process.

#include "gtest/gtest.h"
// TODO(include): confirm ORT test include roots for these headers by grepping
//   onnxruntime/test/providers/*/ for existing OVEP/graph tests, e.g.
//   Grep "qdq_scales_fix|GraphViewer" under targets/onnxruntime/onnxruntime/test.
// #include "core/graph/model.h"
// #include "core/graph/graph_viewer.h"
// #include "core/providers/openvino/qdq_transformations/qdq_scales_fix.h"

namespace onnxruntime {
namespace test {

// TODO(target): confirm the ORT unit-test binary/target that links the OpenVINO
//   provider sources (grep targets/onnxruntime/**/CMakeLists.txt /
//   onnxruntime_unittests.cmake for qdq_scales_fix.cc) — likely
//   onnxruntime_provider_test / onnxruntime_test_all.

TEST(OvepQdqScalesFix, MulWithGraphInputParentDoesNotOobRead) {
  // Build a QDQ uint16 graph in memory:
  //   graph_input X --> Mul (other operand = constant initializer) --> Add
  //   Add path also fed by a Q/DQ pair whose scale/scale_factor > threshold
  //   so scale_graph() triggers apply_scale_to_graph -> down_propagate_scale
  //   -> up_propagate_scale on the Add, recursing up into the Mul, which has
  //   NO DequantizeLinear parent and whose from_node.back() is the graph-input
  //   GraphNode with an empty from_node -> hits qdq_scales_fix.cc:186.
  //
  // TODO(fixture): construct this via onnxruntime::Model + Graph API
  //   (Graph::AddNode for QuantizeLinear/DequantizeLinear/Mul/Add, uint16
  //   scale/zero-point initializers), call graph.Resolve(), obtain a
  //   GraphViewer, then:
  //
  //   std::unique_ptr<onnxruntime::Model> out_model;
  //   auto status = openvino_ep::qdq_scales_fix::Transform(
  //       graph_viewer, DefaultLoggingManager().DefaultLogger(), out_model);
  //
  // Pre-fix expectation (bug present): ASan reports container-overflow / the
  //   process crashes inside up_propagate_scale before returning.
  // Post-fix expectation: no OOB, Transform returns OK and out_model is set.
  //   EXPECT_TRUE(status.IsOK());
  //   EXPECT_NE(out_model, nullptr);
  GTEST_SKIP() << "Skeleton: in-memory QDQ-uint16 model builder + GraphViewer "
                  "wiring for qdq_scales_fix::Transform not yet filled in; "
                  "see TODOs. Encodes the guard for qdq_scales_fix.cc:186.";
}

}  // namespace test
}  // namespace onnxruntime
