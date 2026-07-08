// Agent-authored; NOT compiled or run against the source tree — review before use.
// Encodes the fix for the data race / shared-state mutation at
// targets/onnxruntime/onnxruntime/core/providers/openvino/backend_manager.cc:735-737.
// Pre-fix: BackendManager::Compute() writes the shared SessionContext.device_type="CPU"
// (backend_manager.cc:735) on the NPU->CPU fallback path; because SessionContext is shared
// by reference across all BackendManagers (openvino_execution_provider.cc:143), this both
// races with concurrent reads at backend_manager.cc:700 (TSan data race) and permanently
// corrupts device_type for every other subgraph.
// Post-fix: the fallback must use a local device string for the cache key and a local
// SessionContext copy for MakeBackend, leaving the shared SessionContext.device_type intact.
//
// NOTE: No gtest harness in the unit-test table maps to onnxruntime/core/providers/openvino,
// and the trigger needs a real NPU compile failure + ORT_PARALLEL execution, so this is a
// SKELETON. Fill in the TODOs before use.

#include <gtest/gtest.h>
#include <thread>
#include <string>
// TODO: include the real headers for SessionContext, SubGraphContext, SharedContext,
//       and BackendManager once their include paths are confirmed by reading the repo
//       (core/providers/openvino/contexts.h, backend_manager.h).

TEST(OpenVINOBackendManager, FallbackDoesNotMutateSharedSessionContext) {
  // TODO: construct a SessionContext with device_type = "NPU" and
  //       so_disable_cpu_ep_fallback = false, disable_dynamic_shapes = true.
  // TODO: build a fused ONNX subgraph with a dynamic input shape whose NPU compilation
  //       fails (forces the fallback branch at backend_manager.cc:730-737).
  // TODO: create two BackendManagers sharing the SAME SessionContext& (mirroring
  //       openvino_execution_provider.cc:143).
  //
  // std::string device_before = session_context.device_type;
  // BackendManager bmA(session_context, ...); // subgraph that hits NPU fallback
  // BackendManager bmB(session_context, ...); // independent subgraph
  //
  // Drive bmA.Compute() (fallback) so that any device_type change is observed.
  // ASSERT: the shared SessionContext must be unchanged after the fallback.
  // EXPECT_EQ(session_context.device_type, device_before);  // fails pre-fix (becomes "CPU")
  //
  // Concurrency check (build with -fsanitize=thread): run bmA.Compute() (fallback) and
  // bmB.Compute() (reads device_type at :700) on two threads; TSan must report no data
  // race on SessionContext.device_type once the fix uses a local copy.
  GTEST_SKIP() << "Skeleton: fill in SessionContext/subgraph fixtures and NPU-failure stub.";
}
