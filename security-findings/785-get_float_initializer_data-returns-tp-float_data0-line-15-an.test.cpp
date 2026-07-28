// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for ov_protobuf_utils.cpp:15 / :21 (work item 785).
// The helpers get_float_initializer_data()/set_float_initializer_data() index
// float_data(0) / set_float_data(0,...) after only checking data_type()==FLOAT.
// For a FLOAT TensorProto whose value lives in raw_data (float_data empty), this
// is an out-of-bounds read/write on an empty protobuf RepeatedField (reached from
// qdq_scales_fix.cc:786-788). Pre-fix: ASan heap-buffer-overflow / null-deref, or
// UB. Post-fix (ORT_ENFORCE(float_data_size()>=1) added): throws / handles raw_data.
//
// TODO: confirm the ORT OV EP unit-test target and include paths — ov_protobuf_utils
//       lives in onnxruntime/core/providers/openvino (ORT tree, not an OpenVINO
//       component test dir), so the exact gtest target must be verified by reading
//       onnxruntime's provider test tree before use.
#include "gtest/gtest.h"
#include "core/graph/onnx_protobuf.h"
// TODO: correct relative include for the helper header:
#include "core/providers/openvino/ov_protobuf_utils.h"

using onnxruntime::openvino_ep::get_float_initializer_data;
using onnxruntime::openvino_ep::set_float_initializer_data;

TEST(OvProtobufUtils, FloatInitializerWithEmptyFloatDataIsRejected) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  // 0-dim scalar whose bytes live in raw_data; float_data() is empty.
  float v = 0.5f;
  tp.set_raw_data(&v, sizeof(v));
  ASSERT_EQ(tp.float_data_size(), 0);

  // Pre-fix: float_data(0) reads element 0 of an empty RepeatedField (ASan/UB).
  // Post-fix: guarded (throws) or reads via raw_data.
  EXPECT_ANY_THROW({ (void)get_float_initializer_data(&tp); });
  EXPECT_ANY_THROW({ set_float_initializer_data(&tp, 1.0f); });
}