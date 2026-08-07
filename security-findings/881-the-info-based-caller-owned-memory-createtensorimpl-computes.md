# Security finding #881: The info-based (caller-owned memory) CreateTensorImpl computes stor…

**Full report, discussion and status: https://github.com/ibelem/onnxruntime/issues/61**

| | |
| --- | --- |
| CWE | CWE-843: Type Confusion (leading to CWE-125 OOB read / info leak) |
| Location | [`targets/onnxruntime/onnxruntime/core/session/onnxruntime_c_api.cc:145`](https://github.com/microsoft/onnxruntime/blob/064861b0c019611ba75363c38ad5c8fa7038dfb6/onnxruntime/core/session/onnxruntime_c_api.cc#L145-L167) — `CreateTensorImpl (info-based) / OrtApis::CreateTensorWithDataAsOrtValue()` |
| Trust boundary | ORT public C ABI — host calls CreateTensorWithDataAsOrtValue supplying element type, shape and a raw, caller-owned p_data buffer |
| Validated for | onnxruntime |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/onnxruntime/issues/61 for the root cause, exploit, reproduction and
suggested fix.

A regression test accompanies this report in the same directory.

Models: ideaCreator `claude-4-8-opus`, techLead `claude-4-8-opus`, researcher `claude-4-8-opus`, validator `claude-4-8-opus`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #881.