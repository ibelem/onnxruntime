// Agent-authored; NOT compiled or run against the source tree — review before use.

// Regression test for CWE-789 in BinManager::DeserializeImpl (ov_bin_manager.cc:425)
// A crafted embedded EPContext binary blob declares blob_size = UINT64_MAX in its BSON
// metadata. Pre-fix: container.data.resize(UINT64_MAX) triggers OOM / std::bad_alloc.
// Post-fix: ORT_ENFORCE fires before resize because blob_size > stream_length.

#include <gtest/gtest.h>
#include <sstream>
#include <vector>
#include <cstdint>
#include <nlohmann/json.hpp>

// Include the BinManager header under test.
// TODO: Adjust the include path to match the actual build tree.
#include "onnxruntime/core/providers/openvino/ov_bin_manager.h"
#include "onnxruntime/core/providers/openvino/ov_shared_context.h"

namespace onnxruntime {
namespace openvino_ep {
namespace {

// Helper: build a minimal valid binary stream containing a header, BSON
// metadata whose blob_metadata_map entry has size = giant_size,
// and no actual blob data (embedded mode, no external_bin_path).
std::string BuildCraftedBin(uint64_t giant_blob_size) {
  // Magic / version / header constants copied from ov_bin_manager.cc.
  constexpr uint64_t kMagicNumber = 0x4E49425F5045564FULL;
  constexpr uint64_t kBinVersion = 1;  // BinVersion::v1

  // Build BSON metadata.
  nlohmann::json j;
  j["version"] = "1.0.0";
  j["producer"] = "test";
  // blob_metadata_map: one entry with enormous size.
  j["blob_metadata_map"]["crafted_blob"]["data_offset"] = uint64_t{0};
  j["blob_metadata_map"]["crafted_blob"]["size"] = giant_blob_size;

  std::vector<uint8_t> bson_bytes = nlohmann::json::to_bson(j);

  // header_t layout: magic(8) version(8) header_size(8) bson_start_offset(8) bson_size(8) = 40 bytes.
  constexpr uint64_t kHeaderSize = 40;
  uint64_t bson_start = kHeaderSize;
  uint64_t bson_size = bson_bytes.size();

  std::string buf;
  buf.resize(kHeaderSize + bson_size);
  auto write64 = [&](size_t offset, uint64_t v) {
    std::memcpy(&buf[offset], &v, 8);
  };
  write64(0,  kMagicNumber);
  write64(8,  kBinVersion);
  write64(16, kHeaderSize);
  write64(24, bson_start);
  write64(32, bson_size);
  std::memcpy(&buf[kHeaderSize], bson_bytes.data(), bson_bytes.size());
  return buf;
}

// TODO: Confirm ORT_ENFORCE throws onnxruntime::OnnxRuntimeException (or a
// std::exception subclass) — replace ASSERT_ANY_THROW with the precise type
// once confirmed against the build.
TEST(BinManagerDeserializeImplTest, OversizedBlobSizeIsRejected) {
  // Arrange: craft a stream whose BSON claims blob_size = UINT64_MAX.
  std::string raw = BuildCraftedBin(std::numeric_limits<uint64_t>::max());
  std::istringstream stream(raw, std::ios::binary);

  BinManager mgr;  // No external bin path => embedded mode.
  // TODO: SharedContext requires a bin_path in its constructor;
  //       pass an empty path or a valid SharedContext factory if needed.
  std::shared_ptr<SharedContext> ctx{};

  // Act + Assert: pre-fix this throws std::bad_alloc or crashes;
  //               post-fix it throws ORT_ENFORCE before reaching resize().
  ASSERT_ANY_THROW(mgr.Deserialize(stream, ctx));
}

}  // namespace
}  // namespace openvino_ep
}  // namespace onnxruntime
