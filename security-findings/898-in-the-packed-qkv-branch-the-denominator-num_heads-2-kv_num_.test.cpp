// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-369 divide-by-zero in PagedAttentionTypeAndShapeInference
// (targets/onnxruntime/onnxruntime/core/graph/contrib_ops/bert_defs.cc:1497 and :1500).
// In the packed-QKV branch the denominator (num_heads + 2*kv_num_heads) is computed
// with an unchecked signed int64 multiply. kv_num_heads = INT64_MAX makes 2*kv_num_heads
// wrap to -2; with num_heads = 2 the denominator is 0, so `hidden_size % 0` at :1497
// faults (SIGFPE) during Graph::Resolve. Pre-fix this crashes the process; once the fix
// validates the denominator (denom <= 0 -> fail_shape_inference), Resolve() returns a
// non-OK status instead, which this test asserts.

#include "gtest/gtest.h"

#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "test/test_environment.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

// Builds a com.microsoft PagedAttention node in packed-QKV form (input 2 / value absent,
// kv_cache_layout defaults to SEPARATE) with a 2-D query whose hidden dim is a positive
// value, and pathological head attributes that drive the denominator to zero via signed
// overflow. After the fix, Resolve() must reject the model rather than fault.
TEST(PagedAttentionShapeInferenceTest, PackedQkvHeadOverflowDenominatorRejected) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 17;
  domain_to_version[kMSDomain] = 1;

  Model model("paged_attention_denominator_overflow", /*is_onnx_domain_only=*/false,
              ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, {}, DefaultLoggingManager().DefaultLogger());

  ModelTestBuilder builder(model.MainGraph());
  NodeArg& empty = builder.graph_.GetOrCreateNodeArg("", nullptr);
  // 2-D query, dim[1] positive -> hidden_size = 8 (> 0, so guard at :1495 passes).
  NodeArg* query = builder.MakeInput<float>(std::vector<int64_t>{2, 8});
  NodeArg* output = builder.MakeOutput<float>(std::nullopt);

  // Leave input index 2 (value) empty so the code takes the packed-QKV branch (:1489).
  std::vector<NodeArg*> inputs = {query, &empty, &empty};
  Node& node = builder.AddNode("PagedAttention", inputs, {output}, kMSDomain);
  node.AddAttribute("num_heads", static_cast<int64_t>(2));
  // 2 * INT64_MAX wraps to -2; (2 + -2) == 0 -> divide-by-zero at bert_defs.cc:1497/:1500.
  node.AddAttribute("kv_num_heads", std::numeric_limits<int64_t>::max());
  builder.SetGraphOutputs();

  // Pre-fix: Resolve() faults (SIGFPE) inside shape inference.
  // Post-fix: the zero/negative denominator is rejected and Resolve() returns non-OK.
  Status st = model.MainGraph().Resolve();
  ASSERT_FALSE(st.IsOK()) << "PagedAttention with overflowing head attributes must be rejected, not faulted.";
}

}  // namespace test
}  // namespace onnxruntime
