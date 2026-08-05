// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-125 OOB read at core/graph/graph.cc:6024
// (Graph::FinalizeFuseSubGraph output-edge remap loop). A crafted ORT-format
// model carries a NodeEdge whose output-edge src_arg_index exceeds the node's
// OutputDefs().size(). Pre-fix: Node::LoadEdgesFromOrtFormat (graph.cc:910)
// stores the index unchecked and FinalizeFuseSubGraph indexes OutputDefs with
// it (ASan: heap-buffer-overflow / wild deref via ->Name()). Post-fix: the
// out-of-range arg index is rejected at load time (ORT_RETURN_IF), so the
// session load fails cleanly instead of reading OOB.
//
// SKELETON: the trigger requires a serialized ORT-format (flatbuffers) model
// whose NodeEdge output_edges hold an over-large src_arg_index. That byte
// blob must be produced with the flatbuffers schema builder
// (onnxruntime/core/flatbuffers/schema/ort.fbs), which is why the fixture is
// left as TODO rather than an in-process ov::Model build (this ORT-core path
// has no OpenVINO-API entry point).
#include "gtest/gtest.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

TEST(GraphOrtFormatEdgeValidation, RejectsOutOfRangeOutputEdgeArgIndex) {
  // TODO(blocker: fbs fixture): build an ORT-format model where one node's
  // NodeEdge.output_edges contains an EdgeEnd with src_arg_index >>
  // OutputDefs().size(), using the ort.fbs flatbuffers builder. Emit the bytes
  // as a std::vector<uint8_t> literal here so the crafted input is reviewable.
  std::vector<uint8_t> ort_model_bytes = { /* TODO: crafted ORT-format bytes */ };

  // TODO(helper): register a compiling EP (OpenVINO) via SessionOptions so
  // GetCapability selects the poisoned node into a fused subgraph and
  // Graph::FinalizeFuseSubGraph (graph.cc:6024) is driven.
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "edge-oob");
  Ort::SessionOptions so;

  // Pre-fix: OOB read/wild deref inside FinalizeFuseSubGraph.
  // Post-fix: load fails with an 'Invalid ORT format model' status/throw.
  EXPECT_THROW(
      Ort::Session(env, ort_model_bytes.data(), ort_model_bytes.size(), so),
      Ort::Exception);
}

}  // namespace test
}  // namespace onnxruntime
