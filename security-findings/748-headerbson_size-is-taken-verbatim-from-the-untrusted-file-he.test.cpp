// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Encodes: BinManager::DeserializeImpl (ov_bin_manager.cc:357) must reject a
// header whose bson_size exceeds the actual stream length rather than attempting
// an oversized allocation.  Pre-fix: std::bad_alloc or OOM crash. Post-fix:
// ORT_ENFORCE fires with "BSON size out of bounds" (or equivalent) error.

#include "gtest/gtest.h"
#include <sstream>
#include <cstdint>
#include <cstring>

// TODO: replace with the actual include path once confirmed from the build tree
// #include "core/providers/openvino/ov_bin_manager.h"
// #include "core/providers/openvino/ov_shared_context.h"

namespace onnxruntime {
namespace openvino_ep {
namespace test {

// Mirror of the private header_t in ov_bin_manager.cc — keep in sync.
struct header_t {
  uint64_t magic;         // 0x4E49425F5045564FULL
  uint64_t version;       // 1
  uint64_t header_size;   // sizeof(header_t) == 40
  uint64_t bson_start_offset;
  uint64_t bson_size;
};

static constexpr uint64_t kMagicNumber = 0x4E49425F5045564FULL;
static constexpr uint64_t kVersion     = 1;

// Build a minimal .bin stream: valid header but bson_size >> actual stream bytes.
static std::istringstream BuildCraftedStream(uint64_t bson_size_value) {
  header_t hdr{};
  hdr.magic              = kMagicNumber;
  hdr.version            = kVersion;
  hdr.header_size        = sizeof(header_t);
  hdr.bson_start_offset  = sizeof(header_t); // immediately after header
  hdr.bson_size          = bson_size_value;  // attacker-controlled

  std::string buf(sizeof(header_t), '\0');
  std::memcpy(buf.data(), &hdr, sizeof(hdr));
  // No actual BSON bytes follow — stream ends after the 40-byte header.
  return std::istringstream(buf, std::ios::binary);
}

// TODO: instantiate BinManager with a real (or stub) bin_path once headers are
// confirmed.  Adjust shared_context arg to match the public API.
TEST(BinManagerDeserialize, RejectsBsonSizeLargerThanStream) {
  // bson_size = 2 GiB — vastly exceeds the 40-byte test stream.
  auto stream = BuildCraftedStream(static_cast<uint64_t>(2) * 1024 * 1024 * 1024);

  // TODO: construct BinManager properly; adjust path arg as needed.
  // BinManager mgr(std::filesystem::path{});
  // EXPECT_THROW(mgr.Deserialize(stream, nullptr), onnxruntime::OnnxRuntimeException);
  //
  // Pre-fix behaviour: throws std::bad_alloc (or process OOM) at ov_bin_manager.cc:357.
  // Post-fix behaviour: ORT_ENFORCE fires before the vector<uint8_t> construction.
  //
  // TODO: once headers/symbols are confirmed, remove the stub and uncomment above.
  GTEST_SKIP() << "TODO: wire up BinManager headers and remove this skip";
}

TEST(BinManagerDeserialize, RejectsBsonStartOffsetPlusSizeOverflow) {
  // bson_start_offset == sizeof(header_t), bson_size == UINT64_MAX-sizeof(header_t)+1
  // The sum wraps, exercising the overflow branch of the proposed bounds check.
  uint64_t crafted_size = std::numeric_limits<uint64_t>::max() - sizeof(header_t) + 1;
  auto stream = BuildCraftedStream(crafted_size);

  // TODO: same as above.
  GTEST_SKIP() << "TODO: wire up BinManager headers and remove this skip";
}

}  // namespace test
}  // namespace openvino_ep
}  // namespace onnxruntime