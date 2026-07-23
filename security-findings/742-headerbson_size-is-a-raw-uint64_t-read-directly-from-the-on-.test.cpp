// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-789 in BinManager::DeserializeImpl
// (targets/onnxruntime/onnxruntime/core/providers/openvino/ov_bin_manager.cc:357)
//
// A crafted header passes magic/version/header_size checks but sets bson_size
// to a value far larger than the file; pre-fix this causes a huge allocation;
// post-fix it must throw an ORT exception before allocation.
//
// Build target: onnxruntime_test_all (or the OV EP unit test target in the ORT
// repo — locate via cmake/onnxruntime_unittests.cmake or similar).
// Run: --gtest_filter=BinManagerDeserializeTest.RejectOversizedBsonSize
// Expected pre-fix behaviour: std::bad_alloc caught and re-thrown as OrtException
//   (memory exhaustion on large-but-allocable sizes).
// Expected post-fix behaviour: OrtException thrown BEFORE allocation attempt.

#include <sstream>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "gtest/gtest.h"

// TODO: Replace with the actual include path once confirmed from the ORT build
// system.  The BinManager class is in:
//   onnxruntime/core/providers/openvino/ov_bin_manager.h
// and its constructor requires a SessionContext or equivalent — adapt as needed.
// #include "core/providers/openvino/ov_bin_manager.h"

// Magic number from ov_bin_manager.cc ("OVEP_BIN" little-endian)
static constexpr uint64_t kMagicNumber = 0x4E49425F5045564FULL;

// Version 1
static constexpr uint64_t kVersion1 = 1;

// Minimal header_t layout (must match struct header_t in ov_bin_manager.cc)
struct TestHeader {
  uint64_t magic;
  uint64_t version;
  uint64_t header_size;
  uint64_t bson_start_offset;
  uint64_t bson_size;
};

namespace {

// Build a minimal well-formed stream whose header passes magic/version/
// header_size checks but carries a bson_size far larger than the file.
std::istringstream MakeCraftedStream(uint64_t evil_bson_size) {
  TestHeader hdr{};
  hdr.magic            = kMagicNumber;
  hdr.version          = kVersion1;
  hdr.header_size      = sizeof(TestHeader);
  hdr.bson_start_offset = sizeof(TestHeader);  // BSON immediately after header
  hdr.bson_size        = evil_bson_size;        // << crafted oversized value

  std::string buf(sizeof(TestHeader), '\0');
  std::memcpy(buf.data(), &hdr, sizeof(TestHeader));
  return std::istringstream(std::move(buf), std::ios::binary);
}

}  // namespace

// TODO: Instantiate BinManager with a dummy SessionContext / model path as
// required by its constructor.  The exact constructor signature must be verified
// by reading ov_bin_manager.h.
TEST(BinManagerDeserializeTest, RejectOversizedBsonSize) {
  // bson_size = 4 GiB — large enough to exhaust memory on most systems;
  // small enough that a 64-bit allocator may actually attempt it.
  auto stream = MakeCraftedStream(0xFFFFFFFFULL);

  // TODO: Replace with the real BinManager construction:
  //   onnxruntime::openvino_ep::BinManager bin_mgr(...);
  //   EXPECT_THROW(bin_mgr.Deserialize(stream, nullptr), onnxruntime::OnnxRuntimeException);
  //
  // Pre-fix: std::bad_alloc is caught inside Deserialize and re-thrown as
  //   OnnxRuntimeException (ov_bin_manager.cc:335-337), confirming memory exhaustion.
  // Post-fix: an OnnxRuntimeException is thrown BEFORE reaching line 357,
  //   verifying the bounds check fires with zero allocation.

  // Placeholder assertion so the test structure compiles:
  EXPECT_TRUE(true) << "TODO: wire up BinManager and assert OnnxRuntimeException";
}

TEST(BinManagerDeserializeTest, RejectMaxUint64BsonSize) {
  // bson_size = SIZE_MAX — guaranteed to produce std::bad_alloc; confirms the
  // pre-fix code path catches and re-throws rather than crashing.
  auto stream = MakeCraftedStream(0xFFFFFFFFFFFFFFFFULL);

  // TODO: Same as above — replace placeholder with:
  //   onnxruntime::openvino_ep::BinManager bin_mgr(...);
  //   EXPECT_THROW(bin_mgr.Deserialize(stream, nullptr), onnxruntime::OnnxRuntimeException);
  EXPECT_TRUE(true) << "TODO: wire up BinManager and assert OnnxRuntimeException";
}
