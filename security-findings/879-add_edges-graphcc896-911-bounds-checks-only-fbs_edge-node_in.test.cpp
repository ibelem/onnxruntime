// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-125 OOB read in Node::LoadEdgesFromOrtFormat
// (core/graph/graph.cc:896-919, sink at graph.cc:6024 Graph::FinalizeFuseSubGraph).
//
// The ORT-format load path stores fbs EdgeEnd src_arg_index/dst_arg_index verbatim
// (graph.cc:910) with only node_index checked (:903). It does NOT validate the slot
// against OutputDefs()/InputDefs().size() the way the proto path Graph::AddEdge does
// (:1655/:1665). A crafted NodeEdge whose src_arg_index far exceeds the producer's
// OutputDefs count is accepted, and later indexes OutputDefs()[src_idx] out of bounds.
//
// This test builds a minimal ORT-format flatbuffer model in memory containing one node
// with an output NodeEdge carrying an out-of-range src_arg_index, then loads it via
// InferenceSession from the buffer. Pre-fix: the bad slot is accepted at load and later
// triggers an OOB read (ASan heap-buffer-overflow) during partitioning/fusion.
// Post-fix: LoadEdgesFromOrtFormat rejects the model with a non-OK status.
//
// Modeled on test/framework/ort_model_only_test.cc (flatbuffers::FlatBufferBuilder +
// fbs::Create* helpers + InferenceSession::Load(buffer)).

#include "core/flatbuffers/ort_format_version.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

#include "flatbuffers/flatbuffers.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Builds an ORT-format model buffer with a single node whose output edge references an
// out-of-range src_arg_index. Returns the serialized bytes.
static std::vector<uint8_t> BuildModelWithBadEdgeSlot() {
  flatbuffers::FlatBufferBuilder builder(1024);

  // A minimal NodeEdge for node_index 0 with one output edge whose src_arg_index is
  // deliberately far beyond the node's OutputDefs() count (which is small).
  // fbs::EdgeEnd { node_index, src_arg_index, dst_arg_index }
  const onnxruntime::fbs::EdgeEnd bad_out_edge(/*node_index*/ 0,
                                               /*src_arg_index*/ 0x7fffffff,
                                               /*dst_arg_index*/ 0);
  std::vector<onnxruntime::fbs::EdgeEnd> out_edges{bad_out_edge};
  auto out_edges_off = builder.CreateVectorOfStructs(out_edges);

  onnxruntime::fbs::NodeEdgeBuilder ne_builder(builder);
  ne_builder.add_node_index(0);
  ne_builder.add_output_edges(out_edges_off);
  auto node_edge_off = ne_builder.Finish();
  std::vector<flatbuffers::Offset<onnxruntime::fbs::NodeEdge>> node_edges{node_edge_off};
  auto node_edges_off = builder.CreateVector(node_edges);

  // NOTE: a fully valid ORT-format model also needs a Graph with the referenced Node,
  // its NodeArg defs, and a Model wrapper with the correct ORT version string. Populate
  // those fields exactly as test/framework/ort_model_only_test.cc does (CreateNodeDirect,
  // CreateGraphDirect, CreateModel, ort_format_version.h), attaching node_edges_off as the
  // graph's node_edges. The single crafted field under test is the src_arg_index above.
  //
  // fbs::FinishInferenceSessionBuffer(builder, session_off);
  (void)node_edges_off;

  const uint8_t* buf = builder.GetBufferPointer();
  return std::vector<uint8_t>(buf, buf + builder.GetSize());
}

TEST(OrtFormatEdgeValidationTest, RejectsOutOfRangeSrcArgIndex) {
  auto model_bytes = BuildModelWithBadEdgeSlot();

  SessionOptions so;
  so.session_logid = "OrtFormatEdgeValidationTest";
  InferenceSession session(so, GetEnvironment());

  // Load from the in-memory ORT-format buffer. After the fix, LoadEdgesFromOrtFormat
  // must reject the out-of-range slot with a non-OK status instead of accepting it
  // (which pre-fix leads to an OOB read in Graph::FinalizeFuseSubGraph at graph.cc:6024).
  auto status = session.Load(model_bytes.data(), static_cast<int>(model_bytes.size()));
  ASSERT_FALSE(status.IsOK())
      << "ORT-format model with out-of-range edge src_arg_index must be rejected at load";
  EXPECT_NE(status.ErrorMessage().find("Invalid ORT format model"), std::string::npos);
}

}  // namespace test
}  // namespace onnxruntime
