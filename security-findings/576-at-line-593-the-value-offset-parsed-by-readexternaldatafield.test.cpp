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
