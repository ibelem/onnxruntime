# Security finding #838: scale_graph fetches scale_data = raw_data().data() (L781, via get_m…

**Full report, discussion and status: https://github.com/ibelem/onnxruntime/issues/42**

| | |
| --- | --- |
| CWE | CWE-787: Out-of-bounds Write |
| Location | `targets/onnxruntime/onnxruntime/core/providers/openvino/qdq_transformations/qdq_scales_fix.cc:790` — `scale_graph()` |
| Trust boundary | attacker-controlled DequantizeLinear per-channel scale initializer (declared dims vs raw_data byte length) in a loaded ONNX model, entering via the OpenVINO EP Transform()/scale_graph() graph rewrite |
| Validated for | onnxruntime |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/onnxruntime/issues/42 for the root cause, exploit, reproduction and
suggested fix.

A regression test accompanies this report in the same directory.

Models: ideaCreator `claude-4-8-opus`, techLead `claude-4-8-opus`, researcher `claude-4-8-opus`, validator `claude-4-8-opus`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #838.