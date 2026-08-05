# Security finding #878: In the output-edge remap loop, line 6024 evaluates `node->OutputDef…

**Full report, discussion and status: https://github.com/ibelem/onnxruntime/issues/57**

| | |
| --- | --- |
| CWE | CWE-125: Out-of-bounds Read (vector index + wild pointer deref) |
| Location | `targets/onnxruntime/onnxruntime/core/graph/graph.cc:6024` — `Graph::FinalizeFuseSubGraph()` |
| Trust boundary | ORT public C ABI: attacker-supplied ORT-format model bytes via CreateSessionFromArray / CreateSession, whose per-node output-edge slot indices (fbs_edge->src_arg_index()) are deserialized in Node::LoadEdgesFromOrtFormat and later consumed during EP fused-node replacement. |
| Validated for | onnxruntime |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/onnxruntime/issues/57 for the root cause, exploit, reproduction and
suggested fix.

A regression test accompanies this report in the same directory.

Models: ideaCreator `claude-4-8-opus`, techLead `claude-4-8-opus`, researcher `claude-4-8-opus`, validator `claude-4-8-opus`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #878.