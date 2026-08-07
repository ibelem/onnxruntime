# Security finding #898: In the packed-QKV branch, the denominator `num_heads + 2 * kv_num_h…

**Full report, discussion and status: https://github.com/ibelem/onnxruntime/issues/69**

| | |
| --- | --- |
| CWE | CWE-369: Divide By Zero (via CWE-190 signed integer overflow) |
| Location | [`targets/onnxruntime/onnxruntime/core/graph/contrib_ops/bert_defs.cc:1497`](https://github.com/microsoft/onnxruntime/blob/a86d86ad1bd24e41e919afae2617e41f3184e0de/onnxruntime/core/graph/contrib_ops/bert_defs.cc#L1497-L1501) — `PagedAttentionTypeAndShapeInference()` |
| Trust boundary | Attacker-supplied ONNX graph node attributes (num_heads, kv_num_heads) for a com.microsoft PagedAttention node, loaded via OrtApis::CreateSessionFromArray and processed during Graph::Resolve shape inference. |
| Validated for | onnxruntime |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/onnxruntime/issues/69 for the root cause, exploit, reproduction and
suggested fix.

A regression test accompanies this report in the same directory.

Models: ideaCreator `claude-4-8-opus`, techLead `claude-4-8-opus`, researcher `claude-4-8-opus`, validator `claude-4-8-opus`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #898.