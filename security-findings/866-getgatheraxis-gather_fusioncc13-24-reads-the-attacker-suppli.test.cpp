// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for OOB read at core/optimizer/gather_fusion.cc:200.
// GatherSliceToSplitFusion::ApplyImpl reads shape->dim(axis) with an attacker-controlled
// Gather 'axis' attribute that is never bounds-checked against rank before the access.
// Pre-fix: axis=100 on a rank-2 producer -> shape->dim(100) reads a RepeatedPtrField element
//          past the end (ABSL_DCHECK compiled out in release) -> wild-pointer deref / ASan
//          heap-buffer-overflow READ.
// Post-fix: the `if (axis < 0 || axis >= rank) continue;` guard skips the candidate and the
//           transform completes without touching out-of-range dims (graph unchanged, no throw).
//
// This mirrors the ModelTestBuilder / graph-transformer pattern used in
// onnxruntime/test/optimizer/graph_transform_test.cc.
#include "gtest/gtest.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/optimizer/gather_fusion.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

// Builds: input v (rank 2, shape [4,8]) feeding TWO Gather nodes, each carrying axis=100
// (well past rank=2). The producer's own shape is valid, so shape inference leaves a shape on
// v; the invalid Gather axis is not validated before gather_fusion.cc:200.
TEST(GatherFusionTest, GatherSliceToSplit_OutOfRangeAxis_NoOOBRead) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({4, 8});
    auto* idx0 = builder.MakeScalarInitializer<int64_t>(0);
    auto* idx1 = builder.MakeScalarInitializer<int64_t>(1);
    auto* out0 = builder.MakeOutput();
    auto* out1 = builder.MakeOutput();
    // Produce v via an Identity so it is a single-output node with >=2 Gather consumers.
    auto* v = builder.MakeIntermediate();
    builder.AddNode("Identity", {input}, {v});
    Node& g0 = builder.AddNode("Gather", {v, idx0}, {out0});
    g0.AddAttribute("axis", static_cast<int64_t>(100));  // out-of-range: rank(v) == 2
    Node& g1 = builder.AddNode("Gather", {v, idx1}, {out1});
    g1.AddAttribute("axis", static_cast<int64_t>(100));
  };

  auto model_builder = [&](Model& model) {
    Graph& graph = model.MainGraph();
    ModelTestBuilder helper(graph);
    build_test_case(helper);
    helper.SetGraphOutputs();
    ASSERT_STATUS_OK(graph.Resolve());
  };

  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 13;
  Model model("gather_fusion_oob", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
              DefaultLoggingManager().DefaultLogger());
  model_builder(model);

  Graph& graph = model.MainGraph();
  GraphTransformerManager mgr{5};
  ASSERT_STATUS_OK(mgr.Register(
      std::make_unique<GatherSliceToSplitFusion>(
          InlinedHashSet<std::string_view>{kCpuExecutionProvider}),
      TransformerLevel::Level2));
  // Pre-fix: this Apply performs the OOB read at gather_fusion.cc:200 (ASan aborts here).
  // Post-fix: axis>=rank is skipped, Apply returns OK and leaves the Gather nodes intact.
  ASSERT_STATUS_OK(mgr.ApplyTransformers(graph, TransformerLevel::Level2,
                                         DefaultLoggingManager().DefaultLogger()));

  int gather_count = 0;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Gather") ++gather_count;
  }
  // No fusion should occur for an out-of-range axis; both Gather nodes remain.
  EXPECT_EQ(gather_count, 2);
}

}  // namespace test
}  // namespace onnxruntime
