// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-789 in BinManager::DeserializeImpl
// (targets/onnxruntime/onnxruntime/core/providers/openvino/ov_bin_manager.cc:425)
//
// The flaw: blob_size read from BSON is passed directly to container.data.resize(blob_size)
// with no check that blob_size <= (stream_size - blob_offset).
// Pre-fix: resize(0xFFFFFFFF) (or similar) OOM-kills / throws bad_alloc.
// Post-fix: ORT_ENFORCE fires before the allocation.
//
// Harness: ORT unit tests for the OpenVINO EP.
// TODO: Locate the exact gtest binary name for the ORT OV EP tests.
//       Likely target: onnxruntime_test_providers_openvino or similar.
//       Check CMakeLists.txt under onnxruntime/test/providers/openvino/.

#include "gtest/gtest.h"
#include <sstream>
#include <cstdint>
#include <cstring>

// TODO: Replace with the correct include path for BinManager
// #include "core/providers/openvino/ov_bin_manager.h"
// #include "core/providers/openvino/ov_shared_context.h"

namespace onnxruntime {
namespace openvino_ep {

// Helper: build a minimal valid .bin stream with one blob whose BSON 'size'
// far exceeds the actual stream length.
static std::string BuildMaliciousBin() {
    // TODO: Construct a binary stream matching the header_t layout
    // (magic, version, header_size, bson_start_offset, bson_size)
    // followed by a BSON document whose blob_metadata entry has
    // data_offset=<within stream> and size=0xFFFFFFFF (4 GiB).
    //
    // The BSON can be built with nlohmann::json::to_bson():
    //   nlohmann::json j;
    //   j["version"] = "<current BSONFields::kCurrentBsonVersion>";
    //   j["blob_metadata"]["crafted_blob"]["data_offset"] = <small valid offset>;
    //   j["blob_metadata"]["crafted_blob"]["size"] = (uint64_t)0xFFFFFFFF;
    //   auto bson = nlohmann::json::to_bson(j);
    // Then prepend a header_t pointing to this BSON.
    //
    // TODO: fill in kMagicNumber, BinVersion::current, kCurrentBsonVersion
    //        from ov_bin_manager.h / ov_bin_manager.cc constants.
    return ""; // placeholder
}

TEST(BinManagerDeserializeTest, RejectsOversizedBlobSize) {
    // This test encodes the fix: an oversized blob_size must be rejected
    // before any allocation, not after.
    std::string bin_data = BuildMaliciousBin();
    // TODO: skip test if BuildMaliciousBin() returns empty (not yet implemented).
    if (bin_data.empty()) { GTEST_SKIP() << "Malicious .bin builder not yet implemented — see TODOs."; }

    std::istringstream ss(bin_data);

    // TODO: instantiate BinManager with a dummy path.
    // BinManager mgr("");
    // EXPECT_THROW(mgr.Deserialize(ss, nullptr), onnxruntime::OnnxRuntimeException)
    //     << "Expected ORT_ENFORCE to fire for blob_size exceeding stream bounds";
    //
    // Pre-fix: the above throws std::bad_alloc (or OOM-kills) instead of OnnxRuntimeException.
    // Post-fix: ORT_ENFORCE(blob_size <= stream_size - blob_offset) fires cleanly.
    GTEST_SKIP() << "Skeleton only — see TODOs above for symbol names and binary layout.";
}

} // namespace openvino_ep
} // namespace onnxruntime
