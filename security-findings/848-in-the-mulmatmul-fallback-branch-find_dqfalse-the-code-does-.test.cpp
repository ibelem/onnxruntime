// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-125 OOB read at
//   targets/onnxruntime/onnxruntime/core/providers/openvino/qdq_transformations/qdq_scales_fix.cc:118
//   GraphNode::apply_scale_to_graph, Mul/MatMul find_dq==false branch:
//     from_node.back()->from_node[0]->apply_scale_to_graph(scale_adj)
//
// The test builds an ONNX QDQ subgraph whose QuantizeLinear consumer is a
// DequantizeLinear (so scale_graph enters the scaling path) and whose
// QuantizeLinear's producer is a Mul fed by graph inputs / constant
// initializers so that:
//   - the Mul has NO DequantizeLinear parent (find_dq stays false), and
//   - the parent selected by from_node.back() is a graph-input GraphNode
//     whose from_node vector is empty.
// Pre-fix: line 118 does from_node[0] on an empty vector (ASan
// heap-buffer-overflow / container OOB) and dereferences a wild GraphNode*.
// Post-fix (empty-guard added): Transform completes without OOB.
//
// NOTE: qdq_scales_fix::GraphNode and apply_scale_to_graph are file-internal
// (defined inside qdq_scales_fix.cc, not exported by qdq_scales_fix.h). The
// only reachable entry point is qdq_scales_fix::Transform(const GraphViewer&,
// const Logger&, std::unique_ptr<Model>&). This skeleton drives that entry.

#include "gtest/gtest.h"
#include "core/providers/openvino/qdq_transformations/qdq_scales_fix.h"
#include "core/graph/model.h"
#include "core/graph/graph_viewer.h"
#include "onnx/onnx_pb.h"

using namespace onnxruntime;

TEST(QdqScalesFixTest, MulParentGraphInput_NoOobRead) {
  // TODO(blocker: construct GraphViewer): build an in-memory ONNX model with
  //   input(uint16) -> Mul(input, const_init) -> QuantizeLinear(scale>threshold)
  //                                             -> DequantizeLinear -> output
  // using ONNX_NAMESPACE::ModelProto so that IsQDQGraphWithUint16OrInt16 is
  // true and the Mul has no DequantizeLinear parent (find_dq==false), forcing
  // the qdq_scales_fix.cc:118 branch. Load via onnxruntime::Model::Load and
  // take model->MainGraph() as a GraphViewer.
  // TODO(blocker: logger): obtain a logging::Logger (DefaultLoggingManager).

  // GraphViewer graph_viewer(model->MainGraph());
  // std::unique_ptr<onnxruntime::Model> out_model;
  // Status st = openvino_ep::qdq_scales_fix::Transform(graph_viewer,
  //                                                    logger, out_model);
  // // Pre-fix this line never returns cleanly: ASan reports a
  // // heap-buffer-overflow inside apply_scale_to_graph (from_node[0] on an
  // // empty std::vector) before the wild-pointer method call.
  // ASSERT_TRUE(st.IsOK());
  // ASSERT_NE(out_model, nullptr);
  GTEST_SKIP() << "skeleton: fill in the crafted QDQ ModelProto + GraphViewer/logger construction";
}
