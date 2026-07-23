// Agent-authored; NOT compiled or run against the source tree — review before use.
// Targets: ov_bin_manager.cc:425 — container.data.resize(blob_size) with attacker-controlled blob_size
// from BSON metadata, no upper-bound check against actual stream size.
// Pre-fix: this test causes a bad_alloc or memory exhaustion (resize with huge blob_size).
// Post-fix: ORT_ENFORCE rejects oversized blob_size before resize.

// TODO: Determine actual gtest target for onnxruntime openvino EP unit tests.
// The component lives under onnxruntime/core/providers/openvino/; likely linked
// into onnxruntime_test_utils or a provider-specific test binary. Adjust target name.

#include <gtest/gtest.h>
#include <sstream>
#include <cstdint>
#include <nlohmann/json.hpp>

// TODO: Include BinManager header once the correct include path is confirmed.
// #include "core/providers/openvino/ov_bin_manager.h"

namespace onnxruntime {
namespace openvino_ep {
namespace test {

// Helper: build a minimal valid .bin byte stream with a crafted blob_size
// that exceeds the actual stream content length.
static std::string BuildCraftedBin(uint64_t crafted_blob_size) {
  // TODO: Replicate the header_t layout and BSON structure produced by
  // BinManager::Serialize. Specifically:
  //   1. Write header_t with correct magic, version, header_size, bson_start_offset, bson_size.
  //   2. Write BSON containing BSONFields::kBlobMetadata with one entry:
  //      { "test_blob": { "data_offset": <valid small offset>, "size": crafted_blob_size } }
  //   3. Write a few bytes of actual blob data (much smaller than crafted_blob_size).
  // Use nlohmann::json::to_bson() to produce the BSON bytes.
  // Return the full byte stream as a std::string for use with std::istringstream.
  // TODO: Fill in once header_t definition and BSONFields constants are confirmed from source.
  (void)crafted_blob_size;
  return {};
}

// Test: DeserializeImpl must reject a blob_size that exceeds the actual stream length.
// Pre-fix behavior: resize(crafted_blob_size) with e.g. 0x80000000 bytes on a tiny stream.
// Post-fix behavior: ORT_ENFORCE fires before resize, throwing an onnxruntime::OnnxRuntimeException.
TEST(BinManagerDeserialize, RejectOversizedBlobSizeInEmbeddedStream) {
  // Craft a tiny .bin stream but advertise a blob_size of 2 GB.
  constexpr uint64_t kCraftedBlobSize = 0x80000000ULL;  // 2 GB
  std::string raw = BuildCraftedBin(kCraftedBlobSize);
  // TODO: Remove the skip once BuildCraftedBin is implemented.
  if (raw.empty()) {
    GTEST_SKIP() << "BuildCraftedBin not yet implemented — skeleton only";
  }

  std::istringstream stream(raw, std::ios::binary);

  // TODO: Construct a real BinManager with default options.
  // BinManager manager(/* options */);
  // EXPECT_THROW(manager.Deserialize(stream, nullptr), onnxruntime::OnnxRuntimeException);
  // TODO: Verify the exception message contains "out of bounds" or similar
  // once the fix's ORT_ENFORCE message is known.
}

}  // namespace test
}  // namespace openvino_ep
}  // namespace onnxruntime
