// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-125 OOB read in
//   onnxruntime/core/providers/openvino/qdq_transformations/qdq_scales_fix.cc
// Cited sink: scale_graph() line 724-725  ->  cur_node->to_node[0]->op_type
// (and the identical unguarded access in remove_qdq() line 526-527, which
//  actually trips FIRST via initailize_search()).
//
// The defect: a QuantizeLinear node whose only output is neither consumed by
// another node nor declared as a graph output produces an empty GraphNode::to_node
// vector; to_node[0] then reads past the (null) data pointer and dereferences
// garbage. Pre-fix this crashes (ASan: heap-buffer-overflow / SEGV on unknown
// address). Post-fix (add !to_node.empty() guards) the transform must complete
// without dereferencing an empty vector.
//
// NOTE: scale_graph()/CustomGraph are file-internal (defined in the .cc, not the
// header); the only exported entry is qdq_scales_fix::Transform(GraphViewer,
// logger, model). The test therefore drives the bug through Transform by building
// an in-memory ONNX model containing a dangling uint16 QuantizeLinear.

#include "gtest/gtest.h"
#include "core/graph/model.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/openvino/qdq_transformations/qdq_scales_fix.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

using namespace onnxruntime;

namespace onnxruntime {
namespace test {

// Builds a model with one uint16 QDQ pair (so IsQDQGraphWithUint16OrInt16-style
// preconditions hold) PLUS a QuantizeLinear whose output feeds nothing and is not
// a graph output -> empty to_node in generate_graph_from_onnx.
// TODO(builder): use the ORT test ModelTestBuilder helper to add:
//   - graph input 'x' (float)
//   - initializers 'scale'(float scalar), 'zp'(uint16 scalar)
//   - node Q0 = QuantizeLinear(x, scale, zp) -> 'q_out'      (dangling: no consumer, not an output)
//   - a second valid QDQ pair on a separate branch that IS a graph output
// so the graph resolves but Q0 has no outgoing edge.
static std::unique_ptr<Model> BuildModelWithDanglingQuantizeLinear() {
  // TODO: construct via ModelTestBuilder / GraphAugmenter and return the resolved Model.
  return nullptr;  // TODO: replace with real builder output.
}

TEST(QDQScalesFixTest, DanglingQuantizeLinear_NoOOBRead) {
  auto src_model = BuildModelWithDanglingQuantizeLinear();
  ASSERT_NE(src_model, nullptr);
  GraphViewer src_viewer(src_model->MainGraph());

  const auto& logger = DefaultLoggingManager().DefaultLogger();
  std::unique_ptr<Model> out_model;

  // Pre-fix: qdq_scales_fix::Transform dereferences to_node[0] on an empty vector
  //          (qdq_scales_fix.cc:527 / :725) -> ASan abort / SEGV.
  // Post-fix: the empty-consumer node is skipped and Transform returns OK.
  ASSERT_NO_FATAL_FAILURE({
    Status st = qdq_scales_fix::Transform(src_viewer, logger, out_model);
    ASSERT_TRUE(st.IsOK());
  });
  ASSERT_NE(out_model, nullptr);
}

}  // namespace test
}  // namespace onnxruntime