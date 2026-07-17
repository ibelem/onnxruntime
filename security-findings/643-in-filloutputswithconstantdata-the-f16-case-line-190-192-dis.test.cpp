// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Tests the fix for the heap buffer overflow in FillOutputsWithConstantData
// (backend_utils.cc:190-192) where the f16 case dispatches to
// FillOutputHelper<float>, writing N×4 bytes into an N×2-byte FLOAT16
// ORT output tensor.
//
// Pre-fix: ASan detects a heap-buffer-overflow (WRITE of 2× buffer size).
// Post-fix: f16 case uses a 16-bit type or FillOutputHelper asserts
// sizeof(T) == output element size before std::copy.
//
// File: targets/onnxruntime/onnxruntime/core/providers/openvino/backend_utils.cc:190-192
// Func: FillOutputsWithConstantData → FillOutputHelper<float> (f16 case)

#include "ortAbiTestFixture.h"
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <vector>
#include <cstdint>

using namespace ovep_tests;

// Build an ONNX model with a single FLOAT16 Constant output.
// The model has no inputs — its output is entirely constant-foldable.
// When compiled with OpenVINO EP, CreateOVModel runs ConstantFolding
// (backend_utils.cc:67-68), and all outputs become constants → is_constant=true.
// Then Infer() calls FillOutputsWithConstantData (basic_backend.cc:336).
//
// We craft a minimal ONNX protobuf in-memory:
//   ir_version: 7
//   opset_import: ["" : 17]
//   graph:
//     node: []  (no compute nodes)
//     initializer: [Constant tensor "Y" with FLOAT16 data, shape [4,16]]
//     output: ["Y" : FLOAT16, [4,16]]
//
// This ensures the OV Constant node has element type f16, and the ORT
// output tensor is FLOAT16 (2 bytes/elem). Pre-fix, the 256-byte write
// into the 128-byte buffer triggers ASan.

struct ort_abi_f16_constant_overflow
    : public ortAbiTestFixture,
      public testing::WithParamInterface<std::string> {};

TEST_P(ort_abi_f16_constant_overflow, run) {
  const auto& ov_device_name = GetParam();

  // TODO: Replace with a real crafted .onnx model file path.
  // The model must declare a FLOAT16 output backed by a Constant initializer
  // so that OV constant folding produces an f16 Constant node.
  // A pre-built fixture model should be added to the model library.
  //
  // Expected model: single output "Y" of type FLOAT16, shape [4,16],
  // with an initializer of the same name containing 64 f16 values.
  // No inputs, no compute nodes — fully constant-foldable.
  //
  // const auto& model_info = model_library_t::GetModel("f16_constant_output");
  // network_t network = network_t::create(model_info.model_path_);
  // network.set_cpu_fallback(false);
  //
  // using device_t = ortAbiTestFixture::device_wrapper_t;
  // using graph_t = ortAbiTestFixture::graph_wrapper_t;
  // using pipeline_t = ortAbiTestFixture::pipeline_wrapper_t;
  //
  // device_t device(ov_device_name);
  // graph_t graph(device, network);
  // pipeline_t pipeline(device, graph, {}, {});
  //
  // // Output buffer sized for FLOAT16: 64 elements × 2 bytes = 128 bytes
  // std::vector<uint16_t> output_data(64, 0);
  // std::map<std::string, void*> output_buffers{{"Y", output_data.data()}};
  // pipeline.run({}, output_buffers);
  //
  // // Post-fix: all elements should be the f16 value we set (e.g. 0x3C00 = 1.0)
  // for (size_t i = 0; i < 64; i++) {
  //   EXPECT_EQ(output_data[i], 0x3C00)
  //     << "f16 output element " << i << " corrupted by overflow";
  // }
  //
  // Pre-fix under ASan, the process aborts during pipeline.run() with:
  //   ERROR: AddressSanitizer: heap-buffer-overflow
  //   WRITE of size 256 at ... (buffer size 128)
  //   in FillOutputHelper<float> (backend_utils.cc:208)

  GTEST_SKIP() << "Requires a crafted .onnx fixture model with FLOAT16 constant output. "
    << "See comments above for the expected model structure. "
    << "Add the fixture to model_library_t and uncomment the test body.";
}

INSTANTIATE_TEST_SUITE_P(
    ort_abi_f16_constant_overflow,
    ort_abi_f16_constant_overflow,
    testing::ValuesIn(ortAbiTestFixture::device_names_),
    [](const testing::TestParamInfo<std::string>& info) {
      return info.param;
    });