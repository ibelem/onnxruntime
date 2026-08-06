# Security finding #842: The prepacked loop reads each blob using the model-supplied blob_of…

**Full report, discussion and status: https://github.com/ibelem/onnxruntime/issues/64**

| | |
| --- | --- |
| CWE | CWE-125: Out-of-bounds Read (missing size-consistency validation of attacker-controlled pre-packed blob) |
| Location | [`targets/onnxruntime/onnxruntime/core/framework/tensorprotoutils.cc:1792`](https://github.com/microsoft/onnxruntime/blob/baa0d32bbdf7557772e0678e146de9e0b81c00cc/onnxruntime/core/framework/tensorprotoutils.cc#L1792-L1808) — `GetExtDataFromTensorProto()` |
| Trust boundary | Untrusted ONNX model + its external-data file: the offset/length fields inside a `prepacked_*` external_data string, parsed by ExternalDataInfo::Create and consumed by the framework's pre-packed-weight reconstruction. |
| Validated for | onnxruntime |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/onnxruntime/issues/64 for the root cause, exploit, reproduction and
suggested fix.

A regression test accompanies this report in the same directory.

Models: ideaCreator `claude-4-8-opus`, techLead `claude-4-8-opus`, researcher `claude-4-8-opus`, validator `claude-4-8-opus`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #842.