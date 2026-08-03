// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-787 OOB write in
// onnxruntime/core/providers/openvino/qdq_transformations/qdq_scales_fix.cc:781-792 (scale_graph).
// The scale_graph per-channel branch writes scale_size = product(dims) floats into the
// buffer returned by initializer->raw_data().data(), with no check that
// raw_data().size() >= scale_size*sizeof(float) and no has_raw_data() guard.
// A DequantizeLinear per-channel scale that declares dims=[31] (124 bytes, <=127-byte
// validation gate at graph.cc:3914) but supplies EMPTY raw_data (value carried in float_data)
// yields a zero-capacity destination while scale_size stays 31 -> 124-byte heap OOB write
// under ASan when scale_factor != 1.0f is propagated by the QDQ pass.
//
// Pre-fix: ASan heap-buffer-overflow WRITE inside scale_graph's divide loop.
// Post-fix (bound by raw_data().size()/sizeof(float) + reject when !has_raw_data()):
// Transform() skips/errors on the malformed initializer and no OOB occurs.
//
// SKELETON: qdq_scales_fix::scale_graph is file-internal; the only public entry is
// qdq_scales_fix::Transform(const GraphViewer&, logger, model) declared in
// qdq_scales_fix.h. The test must build an in-memory ONNX model containing a
// QuantizeLinear->DequantizeLinear pair whose DQ scale initializer declares dims=[31]
// as FLOAT but stores its values in float_data (empty raw_data), then drive it through
// a GraphViewer into Transform(). Building that GraphViewer in-process requires ORT graph
// test helpers that are not part of the OpenVINO component test convention table, so the
// harness wiring is left as TODOs below.
#include "gtest/gtest.h"
#include "core/providers/openvino/qdq_transformations/qdq_scales_fix.h"
#include "core/graph/model.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {
namespace test {

TEST(OvepQdqScalesFix, PerChannelScaleShortRawDataNoOob) {
  // TODO(build): confirm the ORT unit-test target that links the OpenVINO EP + qdq_scales_fix.cc
  //              (grep onnxruntime/test CMake for onnxruntime_provider_openvino test target).
  // TODO(construct): build an ONNX_NAMESPACE::ModelProto with:
  //   - input X (uint16), a QuantizeLinear node, then a DequantizeLinear node consuming X;
  //   - the DQ scale initializer 'dq_scale': data_type=FLOAT, dims=[31], float_data with 31
  //     entries and EMPTY raw_data (so raw_data().data() backs a zero-capacity buffer);
  //   - a zero_point initializer so remove_qdq/propagation marks the DQ node scale_factor != 1.0f.
  //   Load via Model::Load(model_proto, model, nullptr, logger) so small (<=127B) initializers
  //   are NOT converted/validated (graph.cc:3914 gate).
  // TODO(drive): GraphViewer viewer(model->MainGraph());
  //              std::unique_ptr<onnxruntime::Model> out;
  //              auto st = openvino_ep::qdq_scales_fix::Transform(viewer, logger, out);
  //   Pre-fix this trips an ASan heap-buffer-overflow WRITE at qdq_scales_fix.cc:791.
  //   Post-fix: EXPECT_TRUE(st.IsOK()) with the scale left untouched, OR EXPECT_FALSE(st.IsOK())
  //   if the fix rejects the malformed initializer — assert no OOB either way.
  GTEST_SKIP() << "wire ORT graph construction helpers per TODOs above";
}

}  // namespace test
}  // namespace onnxruntime
