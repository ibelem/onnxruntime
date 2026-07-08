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
