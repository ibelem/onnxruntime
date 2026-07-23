// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for: ov_bin_manager.cc:412,425
// BinManager::DeserializeImpl does not check blob_offset + blob_size <= file_size
// before container.data.resize(blob_size), allowing attacker-controlled BSON to
// trigger an enormous allocation (CWE-789/CWE-770).
//
// Assertion: DeserializeImpl must throw / enforce-fail when blob_size exceeds
// the stream length, not attempt the allocation.
// Pre-fix: resize(blob_size) is called unconditionally → std::bad_alloc or OOM.
// Post-fix: ORT_ENFORCE(blob_size <= file_size - blob_offset, ...) fires first.

#include <gtest/gtest.h>
#include <sstream>
#include <cstdint>

// TODO: Replace with the actual include path once confirmed.
// #include "core/providers/openvino/ov_bin_manager.h"
// #include "core/providers/openvino/ov_shared_context.h"

namespace onnxruntime {
namespace openvino_ep {
namespace test {

// Helper: build a minimal valid .bin stream whose BSON blob_size is oversized.
// The stream has a valid header + BSON but declares blob_size = 0xFFFFFFFF.
std::istringstream BuildCraftedBinWithOverflowBlobSize() {
  // TODO: Generate a real BSON payload using nlohmann::json BSON encoding,
  // with BSONFields::kBlobMetadata containing:
  //   { "blob0": { kDataOffset: 0, kSize: 4294967295 } }
  // then prepend the BinManager header_t with correct magic, version, bson offsets.
  // This requires reading the exact layout from ov_bin_manager.cc:~30-100 and
  // BSONFields constants — left as TODO for a compiler-verified implementation.
  return std::istringstream(""); // placeholder
}

TEST(BinManagerDeserializeTest, RejectsOversizedBlobSize) {
  // TODO: Instantiate BinManager and SharedContext as done in existing ORT EP tests.
  // Example (adjust to actual API):
  //   auto bin_manager = BinManager(std::nullopt /*no external path*/);
  //   auto shared_ctx = std::make_shared<SharedContext>(...);
  //   auto crafted_stream = BuildCraftedBinWithOverflowBlobSize();
  //
  // Pre-fix: this call exhausts memory or throws std::bad_alloc wrapped in ORT error.
  // Post-fix: ORT_ENFORCE fires with "offset+size out of bounds" before resize.
  //
  // EXPECT_THROW(bin_manager.Deserialize(crafted_stream, shared_ctx),
  //              onnxruntime::OnnxRuntimeException);
  GTEST_SKIP() << "TODO: construct crafted BSON stream with oversized blob_size; "
                  "see ov_bin_manager.cc:412,425";
}

TEST(BinManagerDeserializeTest, RejectsOversizedBlobOffset) {
  // Same as above but kDataOffset = UINT64_MAX, kSize = 1.
  // Pre-fix: seekg(UINT64_MAX) may silently succeed or no-op, then stream.good()
  // check at line 423 may or may not catch it; resize(1) succeeds but read is
  // from wrong position.
  // Post-fix: ORT_ENFORCE(blob_offset <= file_size) fires.
  GTEST_SKIP() << "TODO: construct crafted BSON stream with out-of-bounds blob_offset; "
                  "see ov_bin_manager.cc:411,422";
}

} // namespace test
} // namespace openvino_ep
} // namespace onnxruntime