// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstring>
#include <functional>
#include <mutex>
#include <sstream>
#include <vector>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/common/status.h"
#include "core/common/string_helper.h"
#include "core/framework/allocator.h"
#include "core/framework/callback.h"
#include "core/framework/data_types.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/framework/ort_value.h"
#include "core/framework/plugin_ep_stream.h"
#include "core/framework/tensor.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/TensorSeq.h"
#include "core/framework/utils.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/model_editor_api_types.h"
#include "core/graph/ep_api_types.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/get_execution_providers.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/allocator_adapters.h"
#include "core/session/compile_api.h"
#include "core/session/environment.h"
#include "core/session/ep_graph_assignment_info.h"
#include "core/session/interop_api.h"
#include "core/session/onnxruntime_ep_device_ep_metadata_keys.h"
#include "core/session/plugin_ep/ep_api.h"
#include "core/session/plugin_ep/ep_library_internal.h"
#include "core/session/inference_session.h"
#include "core/session/inference_session_utils.h"
#include "core/session/IOBinding.h"
#include "core/session/lora_adapters.h"
#include "core/session/model_editor_api.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/session/ort_version_check.h"
#include "core/session/utils.h"

#if defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
namespace onnxruntime {
ProviderInfo_CUDA* TryGetProviderInfo_CUDA();
}
#endif

#ifdef ENABLE_TRAINING_APIS
#include "orttraining/training_api/include/onnxruntime_training_c_api.h"
#include "orttraining/training_api/ort_training_apis.h"
#endif

#ifdef USE_CANN
#include "core/providers/cann/cann_provider_factory.h"
#include "core/providers/cann/cann_execution_provider_info.h"
namespace onnxruntime {
ProviderInfo_CANN* TryGetProviderInfo_CANN();
}
#endif

#ifdef USE_DNNL
#include "core/providers/dnnl/dnnl_provider_factory.h"
#include "core/providers/dnnl/dnnl_execution_provider_info.h"
namespace onnxruntime {
ProviderInfo_Dnnl* TryGetProviderInfo_Dnnl();
}
#endif

#ifdef USE_DML
#include "core/providers/dml/dml_provider_factory.h"
const OrtDmlApi* GetOrtDmlApi(_In_ uint32_t version) NO_EXCEPTION;
#endif

#ifdef ENABLE_EXTENSION_CUSTOM_OPS
#include "onnxruntime_extensions.h"
#endif
#if defined(_MSC_VER) && !defined(__clang__)
// The warning is: "Do not assign the result of an allocation or a function call with an owner<T> return value to a raw pointer, use owner<T> instead(i .11)."
// But this file is for C API. It can't use unique_ptr/shared_ptr in function signature.
#pragma warning(disable : 26400)
#endif
using namespace onnxruntime::logging;
using onnxruntime::DataTypeImpl;
using onnxruntime::Environment;
using onnxruntime::IAllocator;
using onnxruntime::InputDefList;
using onnxruntime::narrow;
using onnxruntime::OutputDefList;
using onnxruntime::Tensor;
using onnxruntime::ToOrtStatus;
using onnxruntime::common::Status;

using namespace onnxruntime;

#ifndef ORT_STATUS_PTR
#ifdef _WIN32
#define ORT_STATUS_PTR _Check_return_ _Ret_maybenull_ OrtStatusPtr
#else
#define ORT_STATUS_PTR OrtStatus*
#endif
#endif

#define TENSOR_READ_API_BEGIN                          \
  API_IMPL_BEGIN                                       \
  auto v = reinterpret_cast<const ::OrtValue*>(value); \
  const auto& tensor = v->Get<onnxruntime::Tensor>();

#define TENSOR_READWRITE_API_BEGIN \
  API_IMPL_BEGIN                   \
  auto v = (value);                \
  auto* tensor = v->GetMutable<onnxruntime::Tensor>();

namespace {
// Create tensor. Allocates memory. Tensor owns memory. Allocator is wrapped and stored in a shared_ptr in Tensor.
ORT_STATUS_PTR CreateTensorImpl(MLDataType ml_type, const int64_t* shape, size_t shape_len,
                                OrtAllocator* allocator, OrtValue& value) {
  TensorShape tensor_shape(shape, shape_len);
  AllocatorPtr alloc_ptr = std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(allocator);
  Tensor::InitOrtValue(ml_type, tensor_shape, std::move(alloc_ptr), value);
  return nullptr;
}

// Create Tensor with existing data. Tensor does not own memory.
ORT_STATUS_PTR CreateTensorImpl(MLDataType ml_type,
                                const int64_t* shape, size_t shape_len,
                                const OrtMemoryInfo* info,
                                void* p_data, size_t p_data_len,
                                OrtValue& ort_value) {
  TensorShape tensor_shape(shape, shape_len);
  if (std::any_of(tensor_shape.GetDims().begin(), tensor_shape.GetDims().end(), [](int64_t v) { return v < 0; })) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "tried creating tensor with negative value in shape");
  }
  // Reject string types in pre-allocated memory as CreateSparseTensorWithValuesAsOrtValue does (onnxruntime_c_api.cc:674).
  if (utils::IsDataTypeString(ml_type)) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Can not use strings in pre-allocated memory."
                                 " Use CreateTensorAsOrtValue() to allocate memory inside and copy");
  }

  size_t size_to_allocate = 0;
  Status status = Tensor::CalculateTensorStorageSize(ml_type, tensor_shape, 0 /*alignment*/, size_to_allocate);
  if (!status.IsOK()) {
    return ToOrtStatus(status);
  }
  if (size_to_allocate > p_data_len) {
    std::ostringstream oss;
    oss << "not enough space: expected " << size_to_allocate << ", got " << p_data_len;
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, oss.str().c_str());
  }

  Tensor::InitOrtValue(ml_type, tensor_shape, p_data, *info, ort_value);
  return nullptr;
}

ORT_STATUS_PTR CreateTensorImpl(MLDataType ml_type,
                                const int64_t* shape, size_t shape_len,
                                OrtAllocator* deleter,
                                void* p_data, size_t p_data_len,
                                OrtValue& ort_value) {
  TensorShape tensor_shape(shape, shape_len);
  if (std::any_of(tensor_shape.GetDims().begin(), tensor_shape.GetDims().end(), [](int64_t v) { return v < 0; })) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "tried creating tensor with negative value in shape");
  }

  size_t size_to_allocate = 0;
  Status status = Tensor::CalculateTensorStorageSize(ml_type, tensor_shape, 0 /*alignment*/, size_to_allocate);

  if (!status.IsOK()) {
    return ToOrtStatus(status);
  }

  if (size_to_allocate > p_data_len) {
    std::ostringstream oss;
    oss << "p_data_len was smaller than expected. Expected:" << size_to_allocate << " Got:" << p_data_len;
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, oss.str().c_str());
  }

  AllocatorPtr alloc_ptr = std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(deleter);
  Tensor::InitOrtValue(ml_type, tensor_shape, p_data, std::move(alloc_ptr), ort_value);
  return nullptr;
}

}  // namespace

ORT_API_STATUS_IMPL(OrtApis::CreateEnvWithCustomLogger, OrtLoggingFunction logging_function,
                    _In_opt_ void* logger_param, OrtLoggingLevel logging_level, _In_ const char* logid,
                    _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  OrtEnv::LoggingManagerConstructionInfo lm_info{logging_function, logger_param, logging_level, logid};
  Status status;
  OrtEnvPtr ort_env = OrtEnv::GetOrCreateInstance(lm_info, status);

  *out = ort_env.release();
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateEnv, OrtLoggingLevel logging_level,
                    _In_ const char* logid, _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  OrtEnv::LoggingManagerConstructionInfo lm_info{nullptr, nullptr, logging_level, logid};
  Status status;
  OrtEnvPtr ort_env = OrtEnv::GetOrCreateInstance(lm_info, status);

  *out = ort_env.release();
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateEnvWithGlobalThreadPools, OrtLoggingLevel logging_level,
                    _In_ const char* logid, _In_ const struct OrtThreadingOptions* tp_options, _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  OrtEnv::LoggingManagerConstructionInfo lm_info{nullptr, nullptr, logging_level, logid};
  Status status;
  OrtEnvPtr ort_env = OrtEnv::GetOrCreateInstance(lm_info, status, tp_options);

  *out = ort_env.release();
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateEnvWithCustomLoggerAndGlobalThreadPools, OrtLoggingFunction logging_function, _In_opt_ void* logger_param,
                    OrtLoggingLevel logging_level, _In_ const char* logid, _In_ const struct OrtThreadingOptions* tp_options,
                    _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  OrtEnv::LoggingManagerConstructionInfo lm_info{logging_function, logger_param, logging_level, logid};
  Status status;
  OrtEnvPtr ort_env = OrtEnv::GetOrCreateInstance(lm_info, status, tp_options);

  *out = ort_env.release();
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateEnvWithOptions, _In_ const OrtEnvCreationOptions* options, _Outptr_ OrtEnv** out) {
  API_IMPL_BEGIN
  if (options == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "CreateEnvWithOptions requires a valid (non-null) OrtEnvCreationOptions argument");
  }

  if (out == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "CreateEnvWithOptions requires a valid (non-null) output parameter into which to "
                                 "store the new OrtEnv instance");
  }

  // Both this API function and OrtEnvCreationOptions were added in ORT 1.24, so check that the user
  // filled out the version correctly.
  if (options->version < 24) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "CreateEnvWithOptions requires a OrtEnvCreationOptions argument with the version set "
                                 "equal to ORT_API_VERSION");
  }

  if (options->logging_severity_level < OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE ||
      options->logging_severity_level > OrtLoggingLevel::ORT_LOGGING_LEVEL_FATAL) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "CreateEnvWithOptions requires a OrtEnvCreationOptions argument "
                                 "with a valid logging severity level value from the OrtLoggingLevel enumeration");
  }

  if (options->log_id == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "CreateEnvWithOptions requires a OrtEnvCreationOptions argument "
                                 "with a valid (non-null) log identifier string");
  }

  OrtLoggingLevel logging_severity_level = static_cast<OrtLoggingLevel>(options->logging_severity_level);
  OrtEnv::LoggingManagerConstructionInfo lm_info(options->custom_logging_function,
                                                 options->custom_logging_param,
                                                 logging_severity_level,
                                                 options->log_id);
  Status status;
  OrtEnvPtr ort_env = OrtEnv::GetOrCreateInstance(lm_info, status, options->threading_options, options->config_entries);

  *out = ort_env.release();
  return ToOrtStatus(status);
  API_IMPL_END
}

// enable platform telemetry
ORT_API_STATUS_IMPL(OrtApis::EnableTelemetryEvents, _In_ const OrtEnv* ort_env) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(ort_env);
  // note telemetry is controlled via the platform Env object, not the OrtEnv object instance
  const Env& env = Env::Default();
  env.GetTelemetryProvider().EnableTelemetryEvents();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::DisableTelemetryEvents, _In_ const OrtEnv* ort_env) {
  API_IMPL_BEGIN
  ORT_UNUSED_PARAMETER(ort_env);
  // note telemetry is controlled via the platform Env object, not the OrtEnv object instance
  const Env& env = Env::Default();
  env.GetTelemetryProvider().DisableTelemetryEvents();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UpdateEnvWithCustomLogLevel, _In_ OrtEnv* ort_env,
                    OrtLoggingLevel log_severity_level) {
  API_IMPL_BEGIN
  LoggingManager* default_logging_manager = ort_env->GetLoggingManager();
  int severity_level = static_cast<int>(log_severity_level);
  default_logging_manager->SetDefaultLoggerSeverity(static_cast<logging::Severity>(severity_level));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateTensorWithDataAsOrtValue, _In_ const OrtMemoryInfo* info,
                    _Inout_ void* p_data, size_t p_data_len, _In_ const int64_t* shape, size_t shape_len,
                    ONNXTensorElementDataType type, _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  auto ml_type = DataTypeImpl::TensorTypeFromONNXEnum(type)->GetElementType();
  auto value = std::make_unique<OrtValue>();
  ORT_API_RETURN_IF_ERROR(CreateTensorImpl(ml_type, shape, shape_len, info, p_data, p_data_len, *value));
  *out = value.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateTensorWithDataAndDeleterAsOrtValue, _In_ OrtAllocator* deleter,
                    _In_ void* p_data, size_t p_data_len,
                    _In_ const int64_t* shape, size_t shape_len,
                    ONNXTensorElementDataType type,
                    _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  auto ml_type = DataTypeImpl::TensorTypeFromONNXEnum(type)->GetElementType();
  auto value = std::make_unique<OrtValue>();
  ORT_API_RETURN_IF_ERROR(CreateTensorImpl(ml_type, shape, shape_len, deleter, p_data, p_data_len, *value));
  *out = value.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateTensorAsOrtValue, _Inout_ OrtAllocator* allocator,
                    _In_ const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type,
                    _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
  auto ml_type = DataTypeImpl::TensorTypeFromONNXEnum(type)->GetElementType();
  auto value = std::make_unique<OrtValue>();
  ORT_API_RETURN_IF_ERROR(CreateTensorImpl(ml_type, shape, shape_len, allocator, *value));
  *out = value.release();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateSparseTensorAsOrtValue, _Inout_ OrtAllocator* allocator,
                    _In_ const int64_t* dense_shape,
                    size_t dense_shape_len, ONNXTensorElementDataType type, _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  auto sparse_tensor_type = DataTypeImpl::SparseTensorTypeFromONNXEnum(type);
  auto element_type = sparse_tensor_type->GetElementType();
  assert(element_type->AsPrimitiveDataType() != nullptr);
  TensorShape shape(dense_shape, dense_shape_len);
  if (std::any_of(shape.GetDims().begin(), shape.GetDims().end(),
                  [](int64_t v) { return v < 0; })) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "tried creating tensor with negative value in shape");
  }

  auto alloc_ptr = std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(allocator);
  auto value = std::make_unique<OrtValue>();
  SparseTensor::InitOrtValue(element_type, shape, std::move(alloc_ptr), *value);
  *out = value.release();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(allocator);
  ORT_UNUSED_PARAMETER(dense_shape);
  ORT_UNUSED_PARAMETER(dense_shape_len);
  ORT_UNUSED_PARAMETER(type);
  ORT_UNUSED_PARAMETER(out);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

namespace {
#if !defined(DISABLE_SPARSE_TENSORS)
std::unique_ptr<IDataTransfer> GetDataTransfer(const OrtDevice& src_device, const OrtDevice& dst_device) {
  if (src_device.UsesCpuMemory() && dst_device.UsesCpuMemory()) {
    return std::make_unique<CPUDataTransfer>();
  }

#if defined(USE_CUDA) || defined(USE_CUDA_PROVIDER_INTERFACE)
  if (src_device.Type() == OrtDevice::GPU || dst_device.Type() == OrtDevice::GPU) {
    if (auto* provider_info = TryGetProviderInfo_CUDA()) {
      return provider_info->CreateGPUDataTransfer();
    }
  }
#endif
  ORT_THROW("Not able to find appropriate IDataTransfer to copy sparse data");
}

SparseTensor& ValidateFillInputArgs(OrtValue* v, const TensorShape& values_shape, const OrtMemoryInfo* data_mem_info) {
  auto& sparse_tensor = SparseTensor::GetSparseTensorFromOrtValue(*v);
  if (sparse_tensor.IsDataTypeString()) {
    if (!data_mem_info->device.UsesCpuMemory() || !sparse_tensor.Location().device.UsesCpuMemory()) {
      ORT_THROW("Strings can only reside in CPU memory");
    }
  }
  if (std::any_of(values_shape.GetDims().begin(), values_shape.GetDims().end(),
                  [](int64_t v) { return v < 0; })) {
    ORT_THROW("tried Filling sparse tensor with negative value in values shape");
  }

  return sparse_tensor;
}

union PtrConvert {
  explicit PtrConvert(const void* p_p) : p(p_p) {}
  const void* p;
  const char** strings;
};

// Shared validation for COO indices used by both FillSparseTensorCoo and UseCooIndices.
// Returns nullptr on success, OrtStatus* on validation failure.
OrtStatus* ValidateCooIndices(const int64_t* indices_data, size_t indices_num,
                              size_t values_size, const TensorShape& dense_shape) {
  if (indices_num > 0 && indices_data == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "indices_data must not be null when indices_num > 0.");
  }
  if ((values_size == 0) != (indices_num == 0)) {
    return OrtApis::CreateStatus(
        ORT_INVALID_ARGUMENT,
        values_size == 0
            ? "COO indices must be empty when the sparse tensor has no values."
            : "COO indices must be provided when the sparse tensor has values.");
  }
  if (values_size > 0 && indices_num > 0) {
    if (indices_num == values_size) {
      const auto dense_size = dense_shape.Size();
      for (size_t i = 0; i < indices_num; ++i) {
        if (indices_data[i] < 0 || indices_data[i] >= dense_size) {
          return OrtApis::CreateStatus(
              ORT_INVALID_ARGUMENT,
              MakeString("COO linear index out of bounds: ", indices_data[i],
                         " must be in [0, ", dense_size, ")")
                  .c_str());
        }
      }
    } else if (indices_num / 2 == values_size && indices_num % 2 == 0) {
      if (dense_shape.NumDimensions() != 2) {
        return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                     "COO 2D indices require dense shape of 2 dimensions");
      }
      const auto rows = dense_shape.GetDims()[0];
      const auto cols = dense_shape.GetDims()[1];
      size_t tuple_idx = 0;
      for (size_t i = 0; i < values_size; ++i, tuple_idx += 2) {
        auto r = indices_data[tuple_idx];
        auto c = indices_data[tuple_idx + 1];
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
          return OrtApis::CreateStatus(
              ORT_INVALID_ARGUMENT,
              MakeString("COO 2D index out of bounds: (", r, ", ", c,
                         ") must be in [0, ", rows, ") x [0, ", cols, ")")
                  .c_str());
        }
      }
    } else {
      return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                   "COO indices count must be equal to or twice the values count.");
    }
  }
  return nullptr;
}

// Shared validation for CSR indices used by both FillSparseTensorCsr and UseCsrIndices.
// Returns nullptr on success, OrtStatus* on validation failure.
OrtStatus* ValidateCsrIndices(const int64_t* inner_data, size_t inner_num,
                              const int64_t* outer_data, size_t outer_num,
                              const TensorShape& dense_shape) {
  if (inner_num > 0 && inner_data == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "inner index data must not be null when inner index count > 0.");
  }
  if (outer_num > 0 && outer_data == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "outer index data must not be null when outer index count > 0.");
  }
  if (dense_shape.NumDimensions() == 2 && inner_num > 0) {
    const auto cols = dense_shape.GetDims()[1];
    for (size_t i = 0; i < inner_num; ++i) {
      if (inner_data[i] < 0 || inner_data[i] >= cols) {
        return OrtApis::CreateStatus(
            ORT_INVALID_ARGUMENT,
            MakeString("CSR inner index out of bounds: ", inner_data[i],
                       " must be in [0, ", cols, ")")
                .c_str());
      }
    }
  }
  if (outer_num > 0) {
    if (outer_data[0] != 0) {
      return OrtApis::CreateStatus(
          ORT_INVALID_ARGUMENT,
          MakeString("CSR outer index must start at 0, got: ", outer_data[0]).c_str());
    }
    if (outer_data[outer_num - 1] != static_cast<int64_t>(inner_num)) {
      return OrtApis::CreateStatus(
          ORT_INVALID_ARGUMENT,
          MakeString("CSR outer index last element must equal inner index count (",
                     inner_num, "), got: ", outer_data[outer_num - 1])
              .c_str());
    }
    int64_t prev = 0;
    for (size_t i = 0; i < outer_num; ++i) {
      auto val = outer_data[i];
      if (val < prev || val > static_cast<int64_t>(inner_num)) {
        return OrtApis::CreateStatus(
            ORT_INVALID_ARGUMENT,
            MakeString("CSR outer index out of bounds or not monotonically non-decreasing: ", val).c_str());
      }
      prev = val;
    }
  }
  return nullptr;
}

#endif  // !defined(DISABLE_SPARSE_TENSORS)
}  // namespace

ORT_API_STATUS_IMPL(OrtApis::FillSparseTensorCoo, _Inout_ OrtValue* ort_value, _In_ const OrtMemoryInfo* data_mem_info,
                    _In_ const int64_t* values_shape, size_t values_shape_len, _In_ const void* values,
                    _In_ const int64_t* indices_data, size_t indices_num) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  TensorShape values_t_shape(values_shape, values_shape_len);
  auto& sparse_tensor = ValidateFillInputArgs(ort_value, values_t_shape, data_mem_info);

  auto values_size = narrow<size_t>(values_t_shape.Size());
  auto indices_span = gsl::make_span(indices_data, indices_num);

  if (auto* status = ValidateCooIndices(indices_data, indices_num, values_size,
                                        sparse_tensor.DenseShape())) {
    return status;
  }

  if (sparse_tensor.IsDataTypeString()) {
    PtrConvert conv(values);
    ORT_THROW_IF_ERROR(sparse_tensor.MakeCooStrings(values_size, conv.strings, indices_span));
  } else {
    auto data_transfer = GetDataTransfer(data_mem_info->device, sparse_tensor.Location().device);
    ORT_THROW_IF_ERROR(sparse_tensor.MakeCooData(*data_transfer, *data_mem_info, values_size,
                                                 values, indices_span));
  }
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(data_mem_info);
  ORT_UNUSED_PARAMETER(values_shape);
  ORT_UNUSED_PARAMETER(values_shape_len);
  ORT_UNUSED_PARAMETER(values);
  ORT_UNUSED_PARAMETER(indices_data);
  ORT_UNUSED_PARAMETER(indices_num);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::FillSparseTensorCsr, _Inout_ OrtValue* ort_value, _In_ const OrtMemoryInfo* data_mem_info,
                    _In_ const int64_t* values_shape, size_t values_shape_len, _In_ const void* values,
                    _In_ const int64_t* inner_indices_data, size_t inner_indices_num,
                    _In_ const int64_t* outer_indices_data, size_t outer_indices_num) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  TensorShape values_t_shape(values_shape, values_shape_len);
  auto& sparse_tensor = ValidateFillInputArgs(ort_value, values_t_shape, data_mem_info);
  auto values_size = narrow<size_t>(values_t_shape.Size());

  auto inner_indices_span = gsl::make_span(inner_indices_data, inner_indices_num);
  auto outer_indices_span = gsl::make_span(outer_indices_data, outer_indices_num);

  if (auto* status = ValidateCsrIndices(inner_indices_data, inner_indices_num,
                                        outer_indices_data, outer_indices_num,
                                        sparse_tensor.DenseShape())) {
    return status;
  }

  if (sparse_tensor.IsDataTypeString()) {
    PtrConvert conv(values);
    ORT_THROW_IF_ERROR(sparse_tensor.MakeCsrStrings(values_size, conv.strings, inner_indices_span, outer_indices_span));
  } else {
    auto data_transfer = GetDataTransfer(data_mem_info->device, sparse_tensor.Location().device);
    ORT_THROW_IF_ERROR(sparse_tensor.MakeCsrData(*data_transfer, *data_mem_info, values_size,
                                                 values, inner_indices_span, outer_indices_span));
  }
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(data_mem_info);
  ORT_UNUSED_PARAMETER(values_shape);
  ORT_UNUSED_PARAMETER(values_shape_len);
  ORT_UNUSED_PARAMETER(values);
  ORT_UNUSED_PARAMETER(inner_indices_data);
  ORT_UNUSED_PARAMETER(inner_indices_num);
  ORT_UNUSED_PARAMETER(outer_indices_data);
  ORT_UNUSED_PARAMETER(outer_indices_num);
  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::FillSparseTensorBlockSparse, _Inout_ OrtValue* ort_value, _In_ const OrtMemoryInfo* data_mem_info,
                    _In_ const int64_t* values_shape, size_t values_shape_len, _In_ const void* values,
                    _In_ const int64_t* indices_shape_data, size_t indices_shape_len,
                    _In_ const int32_t* indices_data) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  TensorShape values_t_shape(values_shape, values_shape_len);
  auto& sparse_tensor = ValidateFillInputArgs(ort_value, values_t_shape, data_mem_info);

  TensorShape indices_t_shape(indices_shape_data, indices_shape_len);
  if (std::any_of(indices_t_shape.GetDims().begin(), indices_t_shape.GetDims().end(),
                  [](int64_t v) { return v < 0; })) {
    ORT_THROW("tried Filling sparse tensor with negative value in block sparse indices shape");
  }

  if (sparse_tensor.IsDataTypeString()) {
    PtrConvert conv(values);
    ORT_THROW_IF_ERROR(sparse_tensor.MakeBlockSparseStrings(values_t_shape, conv.strings, indices_t_shape, indices_data));
  } else {
    auto data_transfer = GetDataTransfer(data_mem_info->device, sparse_tensor.Location().device);
    ORT_THROW_IF_ERROR(sparse_tensor.MakeBlockSparseData(*data_transfer, *data_mem_info, values_t_shape,
                                                         values, indices_t_shape, indices_data));
  }
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(data_mem_info);
  ORT_UNUSED_PARAMETER(values_shape);
  ORT_UNUSED_PARAMETER(values_shape_len);
  ORT_UNUSED_PARAMETER(values);
  ORT_UNUSED_PARAMETER(indices_shape_data);
  ORT_UNUSED_PARAMETER(indices_shape_len);
  ORT_UNUSED_PARAMETER(indices_data);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateSparseTensorWithValuesAsOrtValue, _In_ const OrtMemoryInfo* info, _Inout_ void* p_data,
                    _In_ const int64_t* dense_shape, size_t dense_shape_len,
                    _In_ const int64_t* values_shape, size_t values_shape_len,
                    ONNXTensorElementDataType type, _Outptr_ OrtValue** out) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  auto sparse_tensor_type = DataTypeImpl::SparseTensorTypeFromONNXEnum(type);
  auto element_type = sparse_tensor_type->GetElementType();
  assert(element_type->AsPrimitiveDataType() != nullptr);
  if (utils::IsDataTypeString(element_type)) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "Can not use strings in pre-allocated memory."
                                 " Use CreateSparseTensorAsOrtValue() to allocate memory inside and copy");
  }
  TensorShape tensor_dense_shape(dense_shape, dense_shape_len);
  TensorShape tensor_values_shape(values_shape, values_shape_len);
  if (std::any_of(tensor_values_shape.GetDims().begin(), tensor_values_shape.GetDims().end(),
                  [](int64_t v) { return v < 0; })) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "tried creating tensor with negative value in shape");
  }
  auto value = std::make_unique<OrtValue>();
  SparseTensor::InitOrtValue(element_type, tensor_dense_shape, tensor_values_shape, p_data, *info, *value);
  *out = value.release();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(info);
  ORT_UNUSED_PARAMETER(p_data);
  ORT_UNUSED_PARAMETER(dense_shape);
  ORT_UNUSED_PARAMETER(dense_shape_len);
  ORT_UNUSED_PARAMETER(values_shape);
  ORT_UNUSED_PARAMETER(values_shape_len);
  ORT_UNUSED_PARAMETER(type);
  ORT_UNUSED_PARAMETER(out);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UseCooIndices, _Inout_ OrtValue* ort_value, _Inout_ int64_t* indices_data, size_t indices_num) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  auto v = reinterpret_cast<::OrtValue*>(ort_value);
  auto& sparse_tensor = SparseTensor::GetSparseTensorFromOrtValue(*v);
  auto indices_span = (indices_num == 0 || indices_data == nullptr)
                          ? gsl::span<int64_t>()
                          : gsl::make_span(indices_data, indices_num);

  if (auto* status = ValidateCooIndices(indices_data, indices_num,
                                        sparse_tensor.NumValues(),
                                        sparse_tensor.DenseShape())) {
    return status;
  }

  ORT_THROW_IF_ERROR(sparse_tensor.UseCooIndices(indices_span));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(indices_data);
  ORT_UNUSED_PARAMETER(indices_num);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UseCsrIndices, _Inout_ OrtValue* ort_value,
                    _Inout_ int64_t* inner_data, size_t inner_num,
                    _Inout_ int64_t* outer_data, size_t outer_num) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  auto& sparse_tensor = SparseTensor::GetSparseTensorFromOrtValue(*ort_value);
  auto inner_span = (inner_num == 0 || inner_data == nullptr)
                        ? gsl::span<int64_t>()
                        : gsl::make_span(inner_data, inner_num);
  auto outer_span = (outer_num == 0 || outer_data == nullptr)
                        ? gsl::span<int64_t>()
                        : gsl::make_span(outer_data, outer_num);

  if (auto* status = ValidateCsrIndices(inner_data, inner_num, outer_data, outer_num,
                                        sparse_tensor.DenseShape())) {
    return status;
  }

  ORT_THROW_IF_ERROR(sparse_tensor.UseCsrIndices(inner_span, outer_span));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(inner_data);
  ORT_UNUSED_PARAMETER(inner_num);
  ORT_UNUSED_PARAMETER(outer_data);
  ORT_UNUSED_PARAMETER(outer_num);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::UseBlockSparseIndices, _Inout_ OrtValue* ort_value, const int64_t* indices_shape,
                    size_t indices_shape_len, _Inout_ int32_t* indices_data) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  auto& sparse_tensor = SparseTensor::GetSparseTensorFromOrtValue(*ort_value);
  TensorShape ind_shape(indices_shape, indices_shape_len);
  ORT_THROW_IF_ERROR(sparse_tensor.UseBlockSparseIndices(ind_shape, indices_data));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(indices_shape);
  ORT_UNUSED_PARAMETER(indices_shape_len);
  ORT_UNUSED_PARAMETER(indices_data);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetSparseTensorFormat, _In_ const OrtValue* ort_value, _Out_ enum OrtSparseFormat* out) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  auto v = reinterpret_cast<const ::OrtValue*>(ort_value);
  if (!v->IsAllocated()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "the ort_value must contain a constructed tensor");
  }
  const auto& sparse_tensor = v->Get<SparseTensor>();
  *out = static_cast<OrtSparseFormat>(sparse_tensor.Format());
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(out);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetSparseTensorValues, _In_ const OrtValue* ort_value, _Outptr_ const void** out) {
  API_IMPL_BEGIN
#if !defined(DISABLE_SPARSE_TENSORS)
  const auto& sparse_tensor = SparseTensor::GetSparseTensorFromOrtValue(*ort_value);
  if (sparse_tensor.IsDataTypeString()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Use GetStringTensor*() API to retrieve strings");
  }
  const auto& values = sparse_tensor.Values();
  *out = values.DataRaw();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_value);
  ORT_UNUSED_PARAMETER(out);

  return OrtApis::CreateStatus(ORT_FAIL, "SparseTensor is not supported in this build.");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateCustomOpDomain, _In_ const char* domain, _Outptr_ OrtCustomOpDomain** out) {
  API_IMPL_BEGIN
  auto custom_op_domain = std::make_unique<OrtCustomOpDomain>();
  custom_op_domain->domain_ = domain;
  *out = custom_op_domain.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseCustomOpDomain, _Frees_ptr_opt_ OrtCustomOpDomain* ptr) {
  delete ptr;
}

ORT_API_STATUS_IMPL(OrtApis::CustomOpDomain_Add, _Inout_ OrtCustomOpDomain* custom_op_domain, _In_ const OrtCustomOp* op) {
  API_IMPL_BEGIN
  custom_op_domain->custom_ops_.emplace_back(op);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::AddCustomOpDomain, _Inout_ OrtSessionOptions* options,
                    _In_ OrtCustomOpDomain* custom_op_domain) {
  API_IMPL_BEGIN
  options->custom_op_domains_.emplace_back(custom_op_domain);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path, _Outptr_ void** library_handle) {
  API_IMPL_BEGIN

  auto path_str = ToPathString(library_path);
  ORT_API_RETURN_IF_STATUS_NOT_OK(Env::Default().LoadDynamicLibrary(path_str, false, library_handle));
  if (!*library_handle)
    return OrtApis::CreateStatus(ORT_FAIL, "RegisterCustomOpsLibrary: Failed to load library");

  RegisterCustomOpsFn RegisterCustomOps;
  ORT_API_RETURN_IF_STATUS_NOT_OK(Env::Default().GetSymbolFromLibrary(*library_handle, "RegisterCustomOps",
                                                                      (void**)&RegisterCustomOps));
  if (!RegisterCustomOps)
    return OrtApis::CreateStatus(ORT_FAIL, "RegisterCustomOpsLibrary: Entry point RegisterCustomOps not found in library");

  return RegisterCustomOps(options, OrtGetApiBase());
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::RegisterCustomOpsLibrary_V2, _Inout_ OrtSessionOptions* options,
                    _In_ const ORTCHAR_T* library_name) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  ORT_API_RETURN_IF_STATUS_NOT_OK(options->RegisterCustomOpsLibrary(library_name));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(library_name);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Custom operator libraries are not supported in this build");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::RegisterCustomOpsUsingFunction, _Inout_ OrtSessionOptions* options,
                    _In_ const char* registration_func_name) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  if (!registration_func_name) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "RegisterCustomOpsUsingFunction: Registration function name must be specified.");
  }

  RegisterCustomOpsFn RegisterCustomOps;
  ORT_API_RETURN_IF_STATUS_NOT_OK(Env::Default().GetSymbolFromLibrary(nullptr, registration_func_name,
                                                                      (void**)&RegisterCustomOps));
  if (!RegisterCustomOps) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "RegisterCustomOpsUsingFunction: Registration function was not found");
  }

  return RegisterCustomOps(options, OrtGetApiBase());
#else
  ORT_UNUSED_PARAMETER(options);
  ORT_UNUSED_PARAMETER(registration_func_name);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Custom operator libraries are not supported in this build");
#endif
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::EnableOrtCustomOps, _Inout_ OrtSessionOptions* options) {
  API_IMPL_BEGIN

  if (options) {
#ifdef ENABLE_EXTENSION_CUSTOM_OPS
    return RegisterCustomOps(options, OrtGetApiBase());
#else
    return OrtApis::CreateStatus(ORT_FAIL, "EnableOrtCustomOps: Custom operators in onnxruntime-extensions are not enabled");
#endif
  }
  return nullptr;

  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateSession, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  std::unique_ptr<onnxruntime::InferenceSession> sess;
  *out = nullptr;
  ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(options, env, model_path, nullptr, 0, sess));
  ORT_API_RETURN_IF_ERROR(InitializeSession(options, *sess));
  *out = reinterpret_cast<OrtSession*>(sess.release());
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateSessionFromArray, _In_ const OrtEnv* env, _In_ const void* model_data,
                    size_t model_data_length, _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out) {
  API_IMPL_BEGIN
  std::unique_ptr<onnxruntime::InferenceSession> sess;
  ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(options, env, nullptr, model_data, model_data_length, sess));
  ORT_API_RETURN_IF_ERROR(InitializeSession(options, *sess));
  *out = reinterpret_cast<OrtSession*>(sess.release());
  return nullptr;
  API_IMPL_END
}

namespace {
// Checks if there are active lora adapters and adjusts input spans.
void CheckAndAdjustInputSpansForLora(const OrtRunOptions& run_options,
                                     InlinedVector<const char*>& input_names_with_lora,
                                     InlinedVector<const OrtValue*>& inputs_with_lora,
                                     gsl::span<const char* const>& input_names,
                                     gsl::span<const OrtValue* const>& inputs) {
  size_t total_lora_params = 0;
  for (const lora::LoraAdapter* ad : run_options.active_adapters) {
    total_lora_params += ad->GetParamNum();
  }

  input_names_with_lora.reserve(input_names.size() + total_lora_params);
  inputs_with_lora.reserve(inputs.size() + total_lora_params);
  std::copy(input_names.begin(), input_names.end(), std::back_inserter(input_names_with_lora));
  std::copy(inputs.begin(), inputs.end(), std::back_inserter(inputs_with_lora));

  for (const lora::LoraAdapter* ad : run_options.active_adapters) {
    ad->OutputAdapterParameters(std::back_inserter(input_names_with_lora),
                                std::back_inserter(inputs_with_lora));
  }

  input_names = gsl::make_span(input_names_with_lora);
  inputs = gsl::make_span(inputs_with_lora);
}

}  // namespace

ORT_API_STATUS_IMPL(OrtApis::SetEpDynamicOptions, _Inout_ OrtSession* sess,
                    _In_reads_(kv_len) const char* const* keys,
                    _In_reads_(kv_len) const char* const* values,
                    _In_ size_t kv_len) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);

  auto keys_span = gsl::make_span(keys, kv_len);
  auto values_span = gsl::make_span(values, kv_len);

  Status status;

  if (kv_len == 0) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "no inputs were passed");
  } else {
    status = session->SetEpDynamicOptions(keys_span,
                                          values_span);
  }
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::Run, _Inout_ OrtSession* sess, _In_opt_ const OrtRunOptions* run_options,
                    _In_reads_(input_len) const char* const* input_names,
                    _In_reads_(input_len) const OrtValue* const* input, size_t input_len,
                    _In_reads_(output_names_len) const char* const* output_names, size_t output_names_len,
                    _Inout_updates_all_(output_names_len) OrtValue** output) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);

  auto input_names_span = gsl::make_span(input_names, input_len);
  auto input_span = gsl::make_span(input, input_len);
  auto output_name_span = gsl::make_span(output_names, output_names_len);
  auto output_span = gsl::make_span(output, output_names_len);

  Status status;
  if (run_options != nullptr) {
    if (!run_options->active_adapters.empty()) {
      InlinedVector<const char*> input_names_with_lora;
      InlinedVector<const OrtValue*> input_with_lora;

      CheckAndAdjustInputSpansForLora(*run_options, input_names_with_lora, input_with_lora, input_names_span, input_span);

      status = session->Run(*run_options,
                            input_names_span,
                            input_span,
                            output_name_span,
                            output_span);
    } else {
      status = session->Run(*run_options,
                            input_names_span,
                            input_span,
                            output_name_span,
                            output_span);
    }
  } else {
    const RunOptions default_run_options;
    status = session->Run(default_run_options,
                          input_names_span,
                          input_span,
                          output_name_span,
                          output_span);
  }
  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::RunAsync, _Inout_ OrtSession* sess, _In_opt_ const OrtRunOptions* run_options,
                    _In_reads_(input_len) const char* const* input_names,
                    _In_reads_(input_len) const OrtValue* const* input, size_t input_len,
                    _In_reads_(output_names_len) const char* const* output_names, size_t output_names_len,
                    _Inout_updates_all_(output_names_len) OrtValue** output,
                    _In_ RunAsyncCallbackFn run_async_callback, _In_opt_ void* user_data) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);

  if (run_options != nullptr && !run_options->active_adapters.empty()) {
    LOGS(*session->GetLogger(), WARNING) << "RunAsync() active adapters specified, but won't have an effect";
  }

  auto input_names_span = gsl::make_span(input_names, input_len);
  auto input_span = gsl::make_span(input, input_len);
  auto output_name_span = gsl::make_span(output_names, output_names_len);
  auto output_span = gsl::make_span(output, output_names_len);

  return ToOrtStatus(session->RunAsync(run_options,
                                       input_names_span,
                                       input_span,
                                       output_name_span,
                                       output_span,
                                       run_async_callback,
                                       user_data));
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::Session_GetEpGraphAssignmentInfo, _In_ const OrtSession* session,
                    _Outptr_ const OrtEpAssignedSubgraph* const** ep_subgraphs,
                    _Out_ size_t* num_ep_subgraphs) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  const auto* inference_session = reinterpret_cast<const onnxruntime::InferenceSession*>(session);
  const auto& session_options = inference_session->GetSessionOptions();
  bool is_enabled =
      session_options.config_options.GetConfigOrDefault(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "0") == "1";

  if (!is_enabled) {
    std::ostringstream oss;
    oss << "Session configuration entry '" << kOrtSessionOptionsRecordEpGraphAssignmentInfo
        << "' must be set to \"1\" to retrieve EP graph assignment information.";
    return OrtApis::CreateStatus(ORT_FAIL, oss.str().c_str());
  }

  if (ep_subgraphs == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "'ep_subgraphs' argument is null");
  }

  if (num_ep_subgraphs == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "'num_ep_subgraphs' argument is null");
  }

  const std::vector<const OrtEpAssignedSubgraph*>& ep_assignment_info = inference_session->GetEpGraphAssignmentInfo();

  *ep_subgraphs = ep_assignment_info.data();
  *num_ep_subgraphs = ep_assignment_info.size();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(session);
  ORT_UNUSED_PARAMETER(ep_subgraphs);
  ORT_UNUSED_PARAMETER(num_ep_subgraphs);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "EP graph assignment information is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::EpAssignedSubgraph_GetEpName, _In_ const OrtEpAssignedSubgraph* ep_subgraph,
                    _Outptr_ const char** out) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (out == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "EpAssignedSubgraph_GetEpName requires a valid (non-null) `out` output parameter "
                                 "into which to store the EP name string.");
  }

  *out = ep_subgraph->ep_name.c_str();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ep_subgraph);
  ORT_UNUSED_PARAMETER(out);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "EP graph assignment information is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::EpAssignedSubgraph_GetNodes, _In_ const OrtEpAssignedSubgraph* ep_subgraph,
                    _Outptr_ const OrtEpAssignedNode* const** ep_nodes, _Out_ size_t* num_ep_nodes) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (ep_nodes == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "EpAssignedSubgraph_GetNodes requires a valid (non-null) `ep_nodes` output parameter "
                                 "into which to store the pointer to the node array.");
  }

  if (num_ep_nodes == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "EpAssignedSubgraph_GetNodes requires a valid (non-null) `num_ep_nodes` "
                                 "output parameter into which to store the number of nodes.");
  }

  *ep_nodes = ep_subgraph->nodes.data();
  *num_ep_nodes = ep_subgraph->nodes.size();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ep_subgraph);
  ORT_UNUSED_PARAMETER(ep_nodes);
  ORT_UNUSED_PARAMETER(num_ep_nodes);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "EP graph assignment information is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::EpAssignedNode_GetName, _In_ const OrtEpAssignedNode* ep_node,
                    _Outptr_ const char** out) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (out == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "EpAssignedNode_GetName requires a valid (non-null) `out` output parameter "
                                 "into which to store the name string.");
  }

  *out = ep_node->name.c_str();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ep_node);
  ORT_UNUSED_PARAMETER(out);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "EP graph assignment information is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::EpAssignedNode_GetDomain, _In_ const OrtEpAssignedNode* ep_node,
                    _Outptr_ const char** out) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (out == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "EpAssignedNode_GetDomain requires a valid (non-null) `out` output parameter "
                                 "into which to store the domain string.");
  }

  *out = ep_node->domain.c_str();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ep_node);
  ORT_UNUSED_PARAMETER(out);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "EP graph assignment information is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::EpAssignedNode_GetOperatorType, _In_ const OrtEpAssignedNode* ep_node,
                    _Outptr_ const char** out) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (out == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "EpAssignedNode_GetOperatorType requires a valid (non-null) `out` output parameter "
                                 "into which to store the operator type string.");
  }

  *out = ep_node->op_type.c_str();
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ep_node);
  ORT_UNUSED_PARAMETER(out);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "EP graph assignment information is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

struct OrtIoBinding {
  std::unique_ptr<::onnxruntime::IOBinding> binding_;
  explicit OrtIoBinding(std::unique_ptr<::onnxruntime::IOBinding>&& binding) : binding_(std::move(binding)) {}
  OrtIoBinding(const OrtIoBinding&) = delete;
  OrtIoBinding& operator=(const OrtIoBinding&) = delete;
};

ORT_API_STATUS_IMPL(OrtApis::RunWithBinding, _Inout_ OrtSession* sess, _In_ const OrtRunOptions* run_options,
                    _In_ const OrtIoBinding* binding_ptr) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  Status status;
  if (run_options == nullptr) {
    OrtRunOptions default_run_options;
    status = session->Run(default_run_options, *binding_ptr->binding_);
  } else {
    if (!run_options->active_adapters.empty()) {
      LOGS(*session->GetLogger(), WARNING)
          << "RunWithBinding() has active adapters specified, but won't have an effect";
    }
    status = session->Run(*run_options, *binding_ptr->binding_);
  }
  if (!status.IsOK()) {
    return ToOrtStatus(status);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::CreateIoBinding, _Inout_ OrtSession* sess, _Outptr_ OrtIoBinding** out) {
  API_IMPL_BEGIN
  auto session = reinterpret_cast<::onnxruntime::InferenceSession*>(sess);
  std::unique_ptr<::onnxruntime::IOBinding> binding;
  auto status = session->NewIOBinding(&binding);
  if (!status.IsOK()) {
    return ToOrtStatus(status);
  }
  *out = std::make_unique<OrtIoBinding>(std::move(binding)).release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseIoBinding, _Frees_ptr_opt_ OrtIoBinding* binding_ptr) {
  delete binding_ptr;
}

ORT_API_STATUS_IMPL(OrtApis::BindInput, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtValue* val_ptr) {
  API_IMPL_BEGIN
  auto st = binding_ptr->binding_->BindInput(name, *val_ptr);
  if (!st.IsOK()) {
    return ToOrtStatus(st);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::BindOutput, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtValue* val_ptr) {
  API_IMPL_BEGIN
  auto st = binding_ptr->binding_->BindOutput(name, *val_ptr);
  if (!st.IsOK()) {
    return ToOrtStatus(st);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::BindOutputToDevice, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtMemoryInfo* mem_info_ptr) {
  API_IMPL_BEGIN
  auto st = binding_ptr->binding_->BindOutput(name, mem_info_ptr->device);
  if (!st.IsOK()) {
    return ToOrtStatus(st);
  }
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetBoundOutputNames, _In_ const OrtIoBinding* binding_ptr, _In_ OrtAllocator* allocator,
                    _Out_ char** buffer, _Outptr_result_maybenull_ size_t** lengths, _Out_ size_t* count) {
  API_IMPL_BEGIN
  const auto& output_names = binding_ptr->binding_->GetOutputNames();
  if (output_names.empty()) {
    *buffer = nullptr;
    *lengths = nullptr;
    *count = 0U;
    return nullptr;
  }

  IAllocatorUniquePtr<size_t> lengths_alloc(reinterpret_cast<size_t*>(allocator->Alloc(allocator, output_names.size() * sizeof(size_t))),
                                            [allocator](size_t* p) { if(p) allocator->Free(allocator, p); });

  if (!lengths_alloc) {
    return OrtApis::CreateStatus(ORT_FAIL, "lengths allocation failed");
  }

  size_t total_len = 0;
  auto* len_ptr = lengths_alloc.get();
  for (const auto& n : output_names) {
    auto sz = n.size();
    total_len += sz;
    *len_ptr++ = sz;
  }

  IAllocatorUniquePtr<char> buffer_alloc(reinterpret_cast<char*>(allocator->Alloc(allocator, total_len * sizeof(char))),
                                         [allocator](char* p) { if(p) allocator->Free(allocator, p); });

  if (!buffer_alloc) {
    return OrtApis::CreateStatus(ORT_FAIL, "string buffer allocation failed");
  }

  char* buf_ptr = buffer_alloc.get();
  for (const auto& n : output_names) {
    auto sz = n.size();
    memcpy(buf_ptr, n.data(), sz);
    buf_ptr += sz;
  }

  *buffer = buffer_alloc.release();
  *lengths = lengths_alloc.release();
  *count = output_names.size();
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetBoundOutputValues, _In_ const OrtIoBinding* binding_ptr, _In_ OrtAllocator* allocator,
                    _Outptr_result_maybenull_ OrtValue*** output, _Out_ size_t* output_count) {
  API_IMPL_BEGIN
  const auto& outputs
