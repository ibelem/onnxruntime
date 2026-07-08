# Security finding #571: Line 118 calls `utils::ValidateExternalDataPath(blob_filepath, ...)…

**Summary:** Line 118 calls `utils::ValidateExternalDataPath(blob_filepath, ...)…

**CWE IDs:** CWE-367: Time-of-check Time-of-use (TOCTOU) Race Condition
**Severity / Impact:** Arbitrary file read from the process's filesystem context. An attacker-supplied ONNX model can use `ep_cache_context` to name a symlink inside the model directory that, while passing the canonical containment check, is swapped before the `std::ifstream` open to point to any file readable by the ORT/browser process (e.g., `/etc/shadow`, credential files, private keys). This is an information-disclosure vulnerability exploitable from web content via the WebNN API.
**Affected location:** `targets/onnxruntime/onnxruntime/core/providers/openvino/onnx_ctx_model_helper.cc:118` — `EPCtxHandler::GetModelBlobStream()`
**Validated for repos:** onnxruntime
**Trust boundary:** Web content → WebNN API → ORT model load → OpenVINO EP context node attribute ep_cache_context (attacker-controlled ONNX model attribute)

## Description / Root cause
Line 118 calls `utils::ValidateExternalDataPath(blob_filepath, ...)` which internally resolves symlinks via `WeaklyCanonicalPath` and verifies the canonical target is under the model directory. However, lines 119–121 reconstruct the file path from the *raw* (non-canonical) form `blob_filepath.parent_path() / ep_cache_context` and open it with `std::ifstream`. The canonical resolved path is never returned by `ValidateExternalDataPath` and is therefore never used for the actual open. The race window between the resolution/check (line 118) and the open (line 121) allows a symlink-swap attack.

**Validator analysis:** The CWE-367 TOCTOU classification is structurally accurate: ValidateExternalDataPathFromDir (tensorprotoutils.cc:418-461) computes external_data_path_canonical (resolving symlinks) and checks containment/existence, but ValidateExternalDataPath (481-482) discards that canonical result, and GetModelBlobStream reconstructs and opens the non-canonical raw path (onnx_ctx_model_helper.cc:119-121). A symlink whose target is swapped between line 118 and line 121 bypasses the containment check — a genuine confused-deputy symlink race. However, the reported IMPACT and trust boundary are overstated: this is NOT reachable 'from web content via the WebNN API' — WebNN in Chromium constructs models programmatically and does not hand attacker-controlled ONNX EPContext files, cache-file paths, or filesystem symlinks to this loader. Exploitation requires a local attacker with write access to the model directory able to win a narrow race against a higher-privileged ORT process; severity is therefore lower than 'arbitrary file read from web content.' The proposed fix is correct and sufficient in principle: have ValidateExternalDataPath return the resolved canonical path and use ONLY that for exists()/ifstream, plus O_NOFOLLOW on POSIX to eliminate the residual dereference; a robust alternative is to open by fd once and fstat/validate the opened descriptor (check-on-use rather than check-then-use) to fully close the race.

## Exploit / Proof of Concept
1. Place a malicious ONNX EP-context model in a directory the browser/ORT process can load. 2. Set `ep_cache_context` to `legit.bin` where `legit.bin` is initially a symlink pointing to a file inside the model directory (passes `ValidateExternalDataPath`). 3. Win the race: between line 118 returning success and line 121 executing `new std::ifstream(blob_filepath, ...)`, atomically replace `legit.bin` symlink to point to `/etc/shadow` (or any other sensitive file). 4. The `std::ifstream` follows the new symlink, reading the attacker-chosen file. The content is then processed as a model blob stream and can be leaked back via error messages or inferred via side-channel. The race is feasible with a tight spin-loop on a multi-core machine, especially if the model directory is on a slow/network filesystem.

## Reproduction
_(not provided)_

## Test (skeleton)
```cpp
// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression target for CWE-367 TOCTOU in:
//   onnxruntime/core/providers/openvino/onnx_ctx_model_helper.cc:118-121
//   (canonical path from utils::ValidateExternalDataPath is discarded; the raw
//    ep_cache_context path is opened via std::ifstream, allowing a symlink swap
//    between the containment check and the open.)
//
// A true race is not deterministically unit-testable, so this SKELETON instead
// encodes the *fix invariant*: after validation, the path actually opened must be
// the resolved canonical target, and a path that resolves outside the model dir
// must be rejected. Post-fix, ValidateExternalDataPath should surface the
// canonical path (out-param) and GetModelBlobStream must open only that path.
//
// TODO: exact test harness/target for onnxruntime/core/providers/openvino is not
//       in the provided harness table — confirm from the ORT provider test tree
//       (e.g. onnxruntime/test/providers/openvino) before wiring this up.
// TODO: EP_CACHE_CONTEXT-symlink fixture and a minimal EPContext GraphViewer are
//       required to drive GetModelBlobStream end-to-end; provide them or exercise
//       utils::ValidateExternalDataPath directly as below.

#include "gtest/gtest.h"
#include <filesystem>

// TODO: correct include for the util under test.
// #include "core/framework/tensorprotoutils.h"

TEST(OpenVINOEpCtxTOCTOU, ValidatedPathMustBeCanonicalAndContained) {
  namespace fs = std::filesystem;
  // TODO: create a temp model dir with:
  //   model.onnx                (EPContext, embed_mode=0, ep_cache_context="legit.bin")
  //   legit.bin -> <inside-dir target>   (symlink that passes the check)
  // then, after the fix, assert the resolved/opened path equals the canonical
  // target and is contained under the model dir.
  //
  // fs::path canonical;
  // ASSERT_TRUE(onnxruntime::utils::ValidateExternalDataPath(model_path, "legit.bin", &canonical).IsOK());
  // EXPECT_EQ(canonical, fs::weakly_canonical(model_dir / "legit.bin"));
  //
  // Escape case: symlink -> /etc/shadow must be REJECTED even if the name is inside dir.
  // EXPECT_FALSE(onnxruntime::utils::ValidateExternalDataPath(model_path, "escape.bin", &canonical).IsOK());
  GTEST_SKIP() << "Fill in fixtures/out-param per fix; see TODOs.";
}
```
**Build / run:** Build the ORT OpenVINO provider unit test target (confirm exact name in onnxruntime/test/providers/openvino, e.g. onnxruntime_provider_test) and run with --gtest_filter=OpenVINOEpCtxTOCTOU.*. Pre-fix, an equivalent end-to-end test would show ifstream opening the swapped symlink target (path opened != validated canonical path); post-fix the opened path equals the canonical resolved path and out-of-dir symlink targets are rejected. No sanitizer crash is expected (logic/containment assertion, not memory-safety).

## Suggested fix
Modify `ValidateExternalDataPath` (or create a new variant) to return the resolved canonical path to the caller, and use that canonical path for all subsequent file operations. In `GetModelBlobStream`, change:
```
ORT_THROW_IF_ERROR(utils::ValidateExternalDataPath(blob_filepath, std::filesystem::path(ep_cache_context)));
blob_filepath = blob_filepath.parent_path() / ep_cache_context;
```
to:
```
std::filesystem::path canonical_blob_path;
ORT_THROW_IF_ERROR(utils::ValidateExternalDataPath(blob_filepath, std::filesystem::path(ep_cache_context), &canonical_blob_path));
blob_filepath = canonical_blob_path;  // use only the resolved path
```
Then open the file via `blob_filepath` (the canonical form). On POSIX, additionally open with `O_NOFOLLOW` (or via an `open()` fd then `fdopen`) to prevent any remaining symlink dereference.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #571.
