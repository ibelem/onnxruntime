// Agent-authored; NOT compiled or run against the source tree — review before use.
//
// Regression test for CWE-125 OOB read in
//   onnxruntime/core/providers/openvino/ov_stateful_patch_utils.cc:352-353 (FindLLMMatmul)
// The MatMul->Transpose->Result branch does `order = perm_const->get_axis_vector_val();
// slice_gather_dim = order[slice_gather_dim];` with slice_gather_dim==1 by default and no
// bounds check. A Transpose of a rank-1 tensor produces a length-1 perm/order Constant, so
// order[1] reads one size_t past the vector's heap buffer.
//
// Pre-fix: ASan reports heap-buffer-overflow READ inside FindLLMMatmul (or a garbage
//          slice_gather_dim is returned).
// Post-fix (null-check cast + bounds-check slice_gather_dim < order.size()): FindLLMMatmul
//          returns without any OOB access and leaves slice_gather_dim at its default.

#include <memory>
#include <tuple>

#include "gtest/gtest.h"
#include "openvino/openvino.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/result.hpp"

#include "core/providers/openvino/ov_stateful_patch_utils.h"

namespace onnxruntime {
namespace openvino_ep {
namespace test {

// Build: Parameter(rank-1) -> Transpose(perm=[0]) -> Result, and make it model output(0).
// FindLLMMatmul hits the Transpose branch; order.size()==1 while default slice_gather_dim==1.
TEST(OvStatefulPatchUtilsTest, FindLLMMatmul_Rank1TransposeOrder_NoOOBRead) {
  auto param = std::make_shared<ov::op::v0::Parameter>(
      ov::element::f32, ov::PartialShape{ov::Dimension::dynamic()});  // rank 1

  // Transpose "order" input: a length-1 i64 Constant {0} => AxisVector of size 1.
  auto perm = std::make_shared<ov::op::v0::Constant>(
      ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});

  auto transpose = std::make_shared<ov::op::v1::Transpose>(param, perm);
  auto result = std::make_shared<ov::op::v0::Result>(transpose);

  auto model = std::make_shared<ov::Model>(ov::ResultVector{result},
                                           ov::ParameterVector{param});

  // Pre-fix this call performs order[1] (OOB read past a size-1 std::vector) -> ASan abort.
  // Post-fix it must return cleanly; matmul is null (transpose input is a Parameter) and
  // slice_gather_dim must remain the safe default (1) because order is too short to index.
  std::shared_ptr<ov::Node> matmul;
  int64_t slice_gather_dim = -1;
  std::tie(matmul, slice_gather_dim) = FindLLMMatmul(model);

  EXPECT_EQ(matmul, nullptr);
  EXPECT_EQ(slice_gather_dim, 1);  // unchanged default; no OOB-derived garbage
}

}  // namespace test
}  // namespace openvino_ep
}  // namespace onnxruntime
