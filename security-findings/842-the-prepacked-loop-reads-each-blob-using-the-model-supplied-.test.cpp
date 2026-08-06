// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-125 OOB read in prepacked-weight reconstruction.
// Root: core/framework/tensorprotoutils.cc:1792-1808 stores buffer_sizes_ verbatim
// from the model-supplied blob_length with no check that it matches the packed
// size MatMul<float>::PrePack (core/providers/cpu/math/matmul.cc:302-328) produces,
// and MatMul<float>::UseSharedPrePackedBuffers (matmul.cc:330-342) discards the
// size and moves the (undersized) buffer into packed_b_. This test encodes that a
// shared prepacked buffer whose declared size is smaller than the kernel's freshly
// computed packed_b_size must be REJECTED (post-fix) rather than silently accepted
// (pre-fix accepts it, then Compute reads OOB under ASan).
//
// NOTE: onnxruntime's own gtest suite (onnxruntime_test_all) is the correct home
// for this test; the harness enum offered here has no ORT-native token, so this is
// filed as a skeleton against ORT test conventions.
#include "gtest/gtest.h"
#include "core/framework/prepacked_weights.h"
#include "core/framework/allocator.h"
#include "core/providers/cpu/math/matmul.h"

namespace onnxruntime {
namespace test {

// Encodes the missing size-consistency gate. Builds a PrePackedWeights whose
// single buffer is deliberately undersized (16 bytes) relative to the packed_b_size
// a real GemmPackBFp32 over the initializer shape/type would need, then drives it
// through the kernel's UseSharedPrePackedBuffers and asserts the mismatch is caught.
TEST(PrepackedWeightsSizeValidation, RejectsUndersizedSharedBuffer) {
  AllocatorPtr cpu_alloc = std::make_shared<CPUAllocator>();

  // Undersized disk-style blob: 16 bytes, far smaller than any N*K fp32 pack.
  constexpr size_t kUndersized = 16;
  auto buf = IAllocator::MakeUniquePtr<void>(cpu_alloc, kUndersized, true);

  std::vector<BufferUniquePtr> shared_buffers;
  shared_buffers.emplace_back(buf.release(), BufferDeleter(cpu_alloc));
  std::vector<size_t> shared_sizes{kUndersized};

  // TODO(review): instantiate the concrete MatMul<float> kernel via the component's
  // kernel-construction test helper (grep onnxruntime/test for CreateKernel/
  // OpTester MatMul fixtures) with input_idx==1 bound to a large [N,K] fp32
  // initializer so packed_b_size >> kUndersized. Pre-fix, UseSharedPrePackedBuffers
  // returns OK and Compute reads OOB (ASan heap-buffer-overflow). Post-fix, the
  // kernel must compare shared_sizes[0] against its computed packed_b_size and
  // return a non-OK Status.
  //
  // bool used = false;
  // Status st = kernel->UseSharedPrePackedBuffers(shared_buffers, shared_sizes, 1, used);
  // EXPECT_FALSE(st.IsOK()) << "undersized shared prepacked buffer must be rejected";
  SUCCEED() << "skeleton: wire concrete MatMul<float> kernel + [N,K] initializer";
}

}  // namespace test
}  // namespace onnxruntime
