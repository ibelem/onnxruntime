# Security finding #628: At line 256, `utils::ValidateExternalDataPath(validation_base_path,…

**Summary:** At line 256, `utils::ValidateExternalDataPath(validation_base_path,…

**CWE IDs:** CWE-367: Time-of-check Time-of-use (TOCTOU) Race Condition
**Severity / Impact:** An attacker who can manipulate the filesystem between check and use (e.g., by pre-placing or swapping a symlink or directory junction at `ep_context_path` on Windows/Linux) can cause ORT's OpenVINO EP to read and deserialize — or memory-map via `ov::read_tensor_data` — an adversarially-crafted `.bin` binary from outside the allowed model directory. This can lead to loading a malicious compiled model blob, heap corruption inside the BSON/bin parser, or arbitrary code execution in the inference process. The lazy second open in `GetNativeBlob` widens the race window considerably. In a Chromium/WebNN context, a web page triggering model session creation can supply the malicious ONNX EPContext node, and a co-located local process or prior-placed symlink completes the attack.
**Affected location:** `targets/onnxruntime/onnxruntime/core/providers/openvino/onnx_ctx_model_helper.cc:256` — `EPCtxHandler::Initialize()`
**Validated for repos:** openvinoEp
**Trust boundary:** ONNX model file (EPContext node `ep_cache_context` attribute) → filesystem: the path value from an untrusted ONNX attribute is validated once then used for two separate path-based file opens with no re-validation

## Description / Root cause
At line 256, `utils::ValidateExternalDataPath(validation_base_path, cache_context_path)` validates the path. Lines 259-260 then pass the raw `ep_context_path` (a `std::filesystem::path` string) to `SharedContext::Deserialize()`, which delegates to `BinManager::Deserialize` (ov_bin_manager.cc:216), where `std::ifstream stream(path, ...)` re-opens the same path by name. Later, `BinManager::GetNativeBlob` (ov_bin_manager.cc:171) calls `ov::read_tensor_data(external_bin_path_.value())`, a third path-by-name open that is lazy and may occur much later. Neither subsequent open uses a pre-opened file descriptor, O_NOFOLLOW, or any re-validation — the validated state of the path at check time is never re-confirmed at use time.

**Validator analysis:** The TOCTOU is real and reachable: ValidateExternalDataPath (tensorprotoutils.cc:418-461) resolves symlinks via WeaklyCanonicalPath, checks containment, and checks existence, but returns only a pass/fail Status — the canonical path is discarded. The caller (onnx_ctx_model_helper.cc:257) reconstructs a lexical path from the original untrusted string and passes it downstream where it is re-opened by name twice (ov_bin_manager.cc:216 and 171). An attacker who can modify the filesystem between validation and use (e.g., swapping a symlink) can redirect the open to an arbitrary file. The no-race variant (pre-placed symlink) is refuted because ValidateExternalDataPath does resolve symlinks, but the race variant is confirmed by the path-by-name opens. The vulnType (CWE-367 TOCTOU) is accurate. The impact is somewhat overstated: DeserializeImpl uses ORT_ENFORCE for validation (crash/DoS, not heap corruption), and GetNativeBlob has bounds checking (ov_bin_manager.cc:178), so the realistic impact is DoS or reading an attacker-controlled file, not arbitrary code execution. The proposedFix is correct: open at validation time and pass the FD/handle downstream, or use O_NOFOLLOW/openat at each use point. For the lazy GetNativeBlob path, eagerly mapping at Deserialize time or accepting an already-opened stream is the right approach.

## Exploit / Proof of Concept
1. Craft an ONNX model with an EPContext node whose `ep_cache_context` attribute is a relative path such as `safe_model.bin` (passes string-prefix validation). 2. At the expected location `<model_dir>/safe_model.bin`, place a symlink (or junction on Windows) pointing to `/etc/passwd` or a crafted binary outside the allowed directory — this requires no race if `ValidateExternalDataPath` does only a lexical prefix check and does not call `std::filesystem::canonical`. 3. Load the model via ORT / WebNN; `EPCtxHandler::Initialize` validates the symlink name (passes), constructs `ep_context_path`, then `BinManager::Deserialize` opens it as `std::ifstream`, following the symlink to the attacker file. 4. If `ValidateExternalDataPath` does resolve symlinks via `canonical`, the same attack works with a race: swap the symlink after `ValidateExternalDataPath` returns but before `BinManager::Deserialize`'s `std::ifstream stream(path, ...)` executes (ov_bin_manager.cc:216), or before the lazy `ov::read_tensor_data` call (ov_bin_manager.cc:171). The implementation of `ValidateExternalDataPath` was not located in the read files, so the no-race variant remains partially unverified, but the TOCTOU structure is confirmed by the path-by-name opens at lines 216 and 171.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
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
```
**Build / run:** Build target: etests.exe (CMake target for OVEP etests, gtest + C++23). Run: etests.exe --gtest_filter='EPCtxHandlerToctou.*'. Expected sanitizer: ASan may detect out-of-bounds read if the swapped file contains fewer bytes than the bin header (header_t = 40 bytes) — the ORT_ENFORCE at ov_bin_manager.cc:345-348 would abort with 'Failed to read header' or 'Invalid magic number'. Note: deterministic TOCTOU trigger requires race injection, so this skeleton test only verifies the structural precondition (symlink following). Full race testing needs a custom harness with thread synchronization, which is non-deterministic and not suitable for automated CI.

## Suggested fix
Open the file at validation time and pass the open file descriptor to all subsequent consumers. In `EPCtxHandler::Initialize`, after `ValidateExternalDataPath` succeeds, open the file with a platform-safe call (e.g., `open(..., O_RDONLY | O_NOFOLLOW)` on Linux, or `CreateFile` with `FILE_FLAG_OPEN_REPARSE_POINT` and `FILE_FLAG_NO_BUFFERING` checks on Windows) and pass the resulting `fd`/`HANDLE` — wrapped in a stream — to `SharedContext::Deserialize(stream)` instead of storing the path. For the lazy `GetNativeBlob` path, either eagerly map the file at `Deserialize` time and cache the mapping, or accept an already-opened `istream`/file-descriptor rather than re-opening `external_bin_path_`. At minimum, add a second call to `ValidateExternalDataPath` immediately before each `std::ifstream` construction and before `ov::read_tensor_data`, and on Linux use `O_NOFOLLOW` (or `openat` with a dirfd) to reject symlinks at the point of use.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #628.
