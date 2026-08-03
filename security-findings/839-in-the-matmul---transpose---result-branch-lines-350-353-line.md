# Security finding #839: In the `MatMul -> Transpose -> Result` branch (lines 350-353), line…

**Full report, discussion and status: https://github.com/ibelem/onnxruntime/issues/40**

| | |
| --- | --- |
| CWE | CWE-125: Out-of-bounds Read |
| Location | `targets/onnxruntime/onnxruntime/core/providers/openvino/ov_stateful_patch_utils.cc:353` — `FindLLMMatmul()` |
| Trust boundary | Attacker-supplied ONNX model graph loaded by the OpenVINO EP (WinML/ORT) and compiled on the stateful/causal-LM path. |
| Validated for | onnxruntime |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/onnxruntime/issues/40 for the root cause, exploit, reproduction and
suggested fix.

A regression test accompanies this report in the same directory.

Models: ideaCreator `claude-4-8-opus`, techLead `claude-4-8-opus`, researcher `claude-4-8-opus`, validator `claude-4-8-opus`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #839.