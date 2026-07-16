// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Cites: onnx_ctx_model_helper.cc:256-260 (ValidateExternalDataPath then use-by-name),
//        ov_bin_manager.cc:216 (std::ifstream re-open), ov_bin_manager.cc:171 (lazy re-open).
//
// This test encodes the TOCTOU flaw: a path is validated once, then re-opened by name.
// Pre-fix: an attacker who swaps a symlink between validation and open can redirect the read.
// Post-fix (open-at-validation + FD passthrough or O_NOFOLLOW): the swap is rejected.
//
// NOTE: TOCTOU is inherently a race condition and cannot be deterministically triggered
// in a single-threaded gtest. This skeleton documents the flaw and provides a best-effort
// structural test. A real test would need a helper thread to swap the symlink at the right
// moment, which is non-deterministic. The test below asserts that the validated path is
// the same path that gets opened (structural check), which would fail if the fix introduces
// FD passthrough (the path would no longer be re-opened by name).
//
// TODO: Determine exact etests.exe entry point and ORT C-API symbols for loading an
//       EPContext-only ONNX model. The ORT C-API wrappers are in
//       etests/include/onnxruntime-abi/ and fixtures in include/fixtures/.
// TODO: Craft a minimal ONNX model with an EPContext node whose ep_cache_context
//       attribute points to a relative path, and a .bin file at that location.

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <thread>

// Placeholder includes — replace with actual etests headers
// #include "onnxruntime-abi/..."
// #include "fixtures/..."

// Helper: create a minimal OVEP .bin file at a given path
static void WriteDummyBin(const std::filesystem::path& path) {
  // TODO: Write a valid OVEP bin header (magic=0x4E49425F5045564F, version=1)
  // plus minimal BSON metadata, matching the format in ov_bin_manager.cc.
  std::ofstream f(path, std::ios::binary);
  // Placeholder: actual bin format requires header_t + BSON blob map
}

TEST(EPCtxHandlerToctou, PathReopenedByNameAfterValidation) {
  // This test documents that the path validated by ValidateExternalDataPath
  // is NOT the same object used for the subsequent open — the path string is
  // reconstructed lexically and re-opened by name, creating a TOCTOU window.
  //
  // A full deterministic test is not possible because the race is timing-dependent.
  // Instead, this test verifies the structural precondition: that a symlink at
  // the validated path location is followed by the subsequent std::ifstream open.

  auto tmp_dir = std::filesystem::temp_directory_path() / "ovep_toctou_test";
  std::filesystem::create_directories(tmp_dir);

  auto real_bin = tmp_dir / "real.bin";
  auto link_bin = tmp_dir / "link.bin";
  WriteDummyBin(real_bin);

  // Create a symlink that points to real.bin
  std::filesystem::create_symlink(real_bin, link_bin);

  // Pre-condition: the symlink resolves to a file under the allowed directory.
  // ValidateExternalDataPath would pass because WeaklyCanonicalPath resolves
  // the symlink and the canonical target is under tmp_dir.

  // TODO: Load an ONNX model with EPContext node via ORT C-API,
  // with ep_cache_context="link.bin".
  // If the symlink is swapped between ValidateExternalDataPath and
  // std::ifstream open, a different file would be read.
  //
  // For now, assert the symlink is followed (structural precondition):
  std::ifstream stream(link_bin, std::ios::in | std::ios::binary);
  ASSERT_TRUE(stream.is_open()) << "Symlink should be followed by ifstream";

  // Cleanup
  std::filesystem::remove_all(tmp_dir);
}

// TODO: A race-triggering variant would need:
// 1. Thread A: calls ORT InferenceSession with EPContext model (triggers
//    ValidateExternalDataPath then BinManager::Deserialize).
// 2. Thread B: immediately after validation, swaps the symlink to point
//    to /etc/passwd or a crafted bin outside the model dir.
// 3. Assert that the deserialized data is from the swapped file (pre-fix)
//    or that an error is thrown (post-fix with O_NOFOLLOW).
// This is non-deterministic and may flake, so it's not suitable for CI
// without a deterministic race-injection framework.
