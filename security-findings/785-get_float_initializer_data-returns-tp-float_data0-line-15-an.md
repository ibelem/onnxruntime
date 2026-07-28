# Security finding #785: get_float_initializer_data() returns `tp->float_data(0)` (line 15) …

**Full report, discussion and status: https://github.com/ibelem/onnxruntime/issues/37**

| | |
| --- | --- |
| CWE | CWE-125: Out-of-bounds Read / CWE-787: Out-of-bounds Write (via CWE-129 unvalidated repeated-field index) |
| Location | `targets/onnxruntime/onnxruntime/core/providers/openvino/ov_protobuf_utils.cpp:11` — `get_float_initializer_data / set_float_initializer_data()` |
| Trust boundary | WebNN-derived QDQ scale constant initializers flowing into the OpenVINO EP qdq_scales_fix transform (renderer → Mojo → ORT ModelEditor → OV EP graph transform). |
| Validated for | onnxruntime |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/onnxruntime/issues/37 for the root cause, exploit, reproduction and
suggested fix.

A regression test accompanies this report in the same directory.

Models: ideaCreator `claude-4-8-opus`, techLead `claude-4-8-opus`, researcher `claude-4-8-opus`, validator `claude-4-8-opus`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #785.