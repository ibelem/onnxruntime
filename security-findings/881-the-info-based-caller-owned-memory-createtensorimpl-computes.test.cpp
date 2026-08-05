// Agent-authored; NOT compiled or run against the source tree — review before use.
// Regression test for CWE-843 type confusion in the info-based (caller-owned memory)
// CreateTensorImpl at core/session/onnxruntime_c_api.cc:145-168.
//
// Pre-fix: CreateTensorWithDataAsOrtValue accepts ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING
// with a raw caller buffer; Tensor::Init (core/framework/tensor.cc:169) skips
// ConstructStrings because buffer_deleter_ is null, leaving attacker bytes reinterpreted
// as std::string objects. A later GetStringTensorContent (onnxruntime_c_api.cc:1614)
// memcpy's str.data()/str.size() out of those bogus objects -> OOB read / info leak
// (crashes under ASan).
// Post-fix: CreateTensorWithDataAsOrtValue must reject string types for pre-allocated
// memory (mirroring the sparse guard at onnxruntime_c_api.cc:674), so the call throws
// ORT_INVALID_ARGUMENT and the string tensor is never constructed.
//
// Modeled on the ORT shared_lib CApiTest gtest convention (test/shared_lib/*.cc).
#include "core/session/onnxruntime_cxx_api.h"
#include "test/shared_lib/test_fixture.h"
#include <gtest/gtest.h>
#include <array>
#include <cstdint>

extern std::unique_ptr<Ort::Env> ort_env;

TEST(CApiTest, CreateTensorWithData_RejectsStringType) {
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  // N string elements' worth of ATTACKER-CONTROLLED raw bytes. sizeof(std::string)
  // is implementation-defined; oversize the buffer so the :160 size check
  // (which uses elt_type->Size() == sizeof(std::string)) is satisfied for any libc++/libstdc++.
  constexpr size_t kN = 4;
  std::array<uint8_t, kN * 64> raw_bytes;
  raw_bytes.fill(0x41);  // non-empty, non-zero: bogus data()/size() if reinterpreted as std::string

  const std::array<int64_t, 1> shape{static_cast<int64_t>(kN)};

  // Pre-fix this call succeeds and produces a poisoned string OrtValue; the fix makes it throw.
  EXPECT_THROW(
      {
        Ort::Value v = Ort::Value::CreateTensor(
            info, raw_bytes.data(), raw_bytes.size(),
            shape.data(), shape.size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
        (void)v;
      },
      Ort::Exception);
}
