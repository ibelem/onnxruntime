# Security finding #576: At line 593, the value `offset` — parsed by `ReadExternalDataFields…

**Summary:** At line 593, the value `offset` — parsed by `ReadExternalDataFields…

**CWE IDs:** CWE-822: Untrusted Pointer Dereference
**Severity / Impact:** When the model proto produced by `GetModelProtoFromFusedNode` is subsequently consumed (e.g., by `ReadExternalDataForTensor` in tensorprotoutils.cc:216, which executes `memcpy(unpacked_tensor.data(), reinterpret_cast<const void*>(file_offset), tensor_byte_size)`), the attacker-chosen address is dereferenced. This yields an arbitrary read from attacker-chosen process memory (info-leak / ASLR bypass) and, combined with control of `length`, can be escalated to RCE if memory is mapped writeable and the result is forwarded. Affected users: anyone loading an untrusted ONNX model through the WebNN → ORT → OpenVINO EP pipeline when `has_external_weights` is enabled.
**Affected location:** `targets/onnxruntime/onnxruntime/core/providers/openvino/backend_manager.cc:593` — `BackendManager::GetModelProtoFromFusedNode()`
**Validated for repos:** onnxruntime
**Trust boundary:** Attacker-controlled ONNX model file with initializer carrying `data_location=EXTERNAL` and `location="*/_ORT_MEM_ADDR_/*"` (ORT in-memory sentinel) → ORT graph loader → OpenVINO EP initializer re-serialization in GetModelProtoFromFusedNode

## Description / Root cause
At line 593, the value `offset` — parsed by `ReadExternalDataFields` from an attacker-supplied protobuf string via `std::from_chars` at line 419 — is blindly cast to `const void*` and stored as a canonical in-memory pointer: `SetExternalDataFields(proto_init, (const void*)offset, length)`. The guard at line 515/581 (`utils::HasExternalDataInMemory`) only checks that the `location` string equals the sentinel constant (tensorprotoutils.cc:317-319); it performs no validation that the pointer value was placed there by ORT itself. Any ONNX model file can set `location="*/_ORT_MEM_ADDR_/*"` and `offset=<arbitrary integer>`, pass the sentinel check, and have the arbitrary integer stored as the external-data pointer.

**Validator analysis:** The vuln type CWE-822 (Untrusted Pointer Dereference) is accurate for the ORT-core path: a disk-loaded ONNX model whose initializer carries data_location=EXTERNAL, location=_ORT_MEM_ADDR_ and an arbitrary decimal offset survives ORT loading unchecked — ExternalDataInfo::Create (tensor_external_data_info.cc:42-47) and HasExternalDataInMemory (tensorprotoutils.cc:313-325) validate only the location string, never that the offset is an ORT-owned buffer. backend_manager.cc:518/593 re-serializes the attacker integer as a canonical pointer, and ReadExternalDataForTensor (tensorprotoutils.cc:211-227) dereferences it via memcpy, giving arbitrary read (info-leak/ASLR bypass); the RCE escalation claim is speculative. Reachability caveat: line 593 is only reached for OpenVINO >=2025.3 with has_external_weights and >1 in-memory external initializer (backend_manager.cc:512,528-530); the raw memcpy is also independently reachable in ORT core for any untrusted disk model. Not reachable from the WebNN web boundary (offsets there are genuine ORT-owned pointers). The proposed fix is correct in direction but the most robust and sufficient minimal fix is fix-(b)/its final clause: reject any TensorProto loaded from disk whose external-data location equals kTensorProtoLittleEndianMemoryAddressTag/kTensorProtoNativeEndianMemoryAddressTag early (in ExternalDataInfo::Create / model load), since the in-memory sentinel must only ever be produced internally by ORT with a real pointer — a whitelist of registered (ptr,length) ranges is a stronger but heavier alternative.

## Exploit / Proof of Concept
1. Craft an ONNX model file with one initializer (tensor_proto) where `data_location=EXTERNAL`, `external_data["location"]="*/_ORT_MEM_ADDR_/*"`, `external_data["offset"]="<target address as decimal>"`(e.g., `"140737488355328"` for a known library base), and `external_data["length"]="4096"`. 2. Submit this model via the WebNN API (or directly to an ORT session with the OpenVINO EP and `has_external_weights=true`). 3. `HasExternalDataInMemory` returns `true` (only checks the location string). 4. `ReadExternalDataFields` parses the attacker-supplied decimal offset into `size_t offset` at line 419 with no bounds check. 5. Line 593 casts it to `const void*`. 6. Downstream code in `ReadExternalDataForTensor` (tensorprotoutils.cc:216) executes `memcpy(dst, reinterpret_cast<const void*>(file_offset), tensor_byte_size)`, reading from the attacker-chosen address.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-822 in ORT OpenVINO EP path.
// Encodes the fix for:
//   core/providers/openvino/backend_manager.cc:593  (offset cast to const void*)
//   core/framework/tensorprotoutils.cc:216          (memcpy from reinterpret_cast<const void*>(file_offset))
//   core/framework/tensor_external_data_info.cc:42   (Create accepts _ORT_MEM_ADDR_ sentinel from disk)
// Pre-fix: loading a disk model whose initializer has external_data location="_ORT_MEM_ADDR_"
//         and offset=<arbitrary integer> is accepted, and the arbitrary integer is later
//         dereferenced (ASan: SEGV / heap-buffer-overflow / wild pointer read).
// Post-fix: model load must FAIL with a status error because the in-memory sentinel location
//         is not permitted for on-disk TensorProtos.
//
// NOTE: This requires a crafted .onnx fixture and a full ORT session; it cannot be a pure
// self-contained unit without a binary model file, so this is a SKELETON.

#include "gtest/gtest.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_external_data_info.h"
#include "onnx/onnx_pb.h"

namespace onnxruntime {
namespace test {

// TODO: confirm exact test target/namespace by reading onnxruntime/test/framework/*.
TEST(TensorProtoUtilsSecurityTest, RejectsInMemorySentinelFromDiskModel) {
  // TODO: Build a TensorProto as if parsed from an attacker-supplied on-disk model:
  //   data_location = EXTERNAL
  //   external_data["location"] = "_ORT_MEM_ADDR_"   (kTensorProtoNativeEndianMemoryAddressTag)
  //   external_data["offset"]   = "140737488355328" (arbitrary address)
  //   external_data["length"]   = "4096"
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_name("w");
  tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  tp.add_dims(1024);
  tp.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
  auto* loc = tp.add_external_data(); loc->set_key("location");
  // TODO: use the real constant kTensorProtoNativeEndianMemoryAddressTag (UTF-8 form).
  loc->set_value("_ORT_MEM_ADDR_");
  auto* off = tp.add_external_data(); off->set_key("offset"); off->set_value("140737488355328");
  auto* len = tp.add_external_data(); len->set_key("length"); len->set_value("4096");

  // Post-fix expectation: ExternalDataInfo::Create (or an on-load validator) rejects the
  // sentinel location for a proto that originates from disk.
  std::unique_ptr<onnxruntime::ExternalDataInfo> info;
  Status st = onnxruntime::ExternalDataInfo::Create(tp.external_data(), info);
  // TODO: the fix must make Create (called on a disk-origin proto) fail; assert accordingly.
  EXPECT_FALSE(st.IsOK());

  // And the read path must never memcpy from the raw integer.
  std::vector<uint8_t> out;
  Status rd = onnxruntime::utils::ReadExternalDataForTensor(tp, std::filesystem::path{}, out);
  EXPECT_FALSE(rd.IsOK());
}

}  // namespace test
}  // namespace onnxruntime
```
**Build / run:** Build target: onnxruntime_test_all (ORT gtest). Run: onnxruntime_test_all --gtest_filter=TensorProtoUtilsSecurityTest.RejectsInMemorySentinelFromDiskModel . Build with -fsanitize=address. Pre-fix expected: ASan reports SEGV / wild-pointer read inside ReadExternalDataForTensor (tensorprotoutils.cc:216 memcpy from reinterpret_cast<const void*>(file_offset)); post-fix expected: both Create() and ReadExternalDataForTensor() return non-OK Status and the test passes with no ASan report. TODO: replace inline TensorProto construction with a checked-in crafted .onnx fixture if the harness requires full session load through the OpenVINO EP (session_context_.has_external_weights=true, >1 in-memory external initializer, OpenVINO>=2025.3).

## Suggested fix
In `GetModelProtoFromFusedNode` (and in `ReadExternalDataForTensor`), the parsed `offset` must be validated against a whitelist of ORT-owned allocation ranges before use as a pointer. The minimal fix: (a) In `GetModelProtoFromFusedNode` at line 581-593, only enter the `HasExternalDataInMemory` branch if `src_init` was explicitly registered by ORT's initializer-population path (e.g., track `(ptr, length)` pairs in a session-scoped set when `SetExternalDataFields` is first called). (b) In `ReadExternalDataForTensor` (tensorprotoutils.cc:211-228), before the `memcpy`, validate `file_offset` against a registered set of ORT-owned buffers. At a minimum, add a compile-time policy that models loaded from disk (i.e., not constructed in-process by ORT) are NEVER allowed to carry the in-memory sentinel location string — reject any on-disk TensorProto whose `location` equals `kTensorProtoLittleEndianMemoryAddressTag` or `kTensorProtoNativeEndianMemoryAddressTag` with an error status early in model loading (e.g., in `ExternalDataInfo::Create` at tensor_external_data_info.cc:42-43, treat sentinel strings as invalid location values).


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #576.
