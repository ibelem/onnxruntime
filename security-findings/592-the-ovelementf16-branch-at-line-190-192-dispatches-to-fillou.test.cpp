// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for WI-592: backend_utils.cc:190-208.
// Pre-fix: the ov::element::f16 branch dispatches FillOutputHelper<float>, which
// cast_vector<float>()s N f16 constants into N 4-byte floats and std::copy's them
// into an N*2-byte FLOAT16 ORT output buffer -> ASan heap-buffer-overflow.
// Post-fix (f16 branch writes raw 16-bit data / uses <uint16_t>): the copy is
// bounded to N*2 bytes and the assertion below on output byte size holds.
//
// SKELETON: the in-tree ORT OpenVINO EP has no simple standalone gtest entry in
// this repo's provided harness table, and triggering the constant-output path
// needs a crafted ONNX subgraph whose FLOAT16 output const-folds to an
// ov::op::v0::Constant. Fill in the TODOs against the ORT EP test tree.
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
// TODO: include the ORT session/OpenVINO EP test fixtures used by
//       onnxruntime/test/providers/openvino/*.

TEST(OpenVINO_EP_ConstantOutput, F16ConstantOutputDoesNotOverflow) {
  // TODO: load/construct an ONNX model with:
  //   - a FLOAT16 graph output
  //   - value produced ONLY from FLOAT16 initializers (e.g. Add(constA,constB))
  //     so OpenVINO const-folds it into a Constant (populates const_outputs_map).
  // const char* model_path = "crafted_f16_const_output.onnx"; // TODO fixture
  // Ort::SessionOptions so; so.AppendExecutionProvider_OpenVINO_V2({{"device_type","CPU"}});
  // Ort::Session session(env, model_path, so);
  //
  // Run inference; a correct fix writes exactly N*sizeof(uint16_t) bytes.
  // ASSERT_NO_FATAL_FAILURE(session.Run(...));  // pre-fix: ASan abort here
  //
  // Optionally verify the produced output element type/size:
  // auto info = out.GetTensorTypeAndShapeInfo();
  // EXPECT_EQ(info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
  // EXPECT_EQ(out.GetTensorSizeInBytes(), N * 2u);
  GTEST_SKIP() << "Provide crafted f16-constant-output .onnx fixture (see TODOs).";
}
