# Security finding #879: add_edges (graph.cc:896-911) bounds-checks only fbs_edge->node_inde…

**Full report, discussion and status: https://github.com/ibelem/onnxruntime/issues/59**

| | |
| --- | --- |
| CWE | CWE-125: Out-of-bounds Read |
| Location | `targets/onnxruntime/onnxruntime/core/graph/graph.cc:910` — `Node::LoadEdgesFromOrtFormat()` |
| Trust boundary | ORT C ABI: attacker-supplied ORT-format (flatbuffer) model bytes loaded via OrtApis::CreateSessionFromArray (onnxruntime_c_api.cc:933), which sniffs and accepts the ORT flatbuffer format. |
| Validated for | onnxruntime |

This file is an index entry only — it deliberately does not duplicate the
issue. See https://github.com/ibelem/onnxruntime/issues/59 for the root cause, exploit, reproduction and
suggested fix.

A regression test accompanies this report in the same directory.

Models: ideaCreator `claude-4-8-opus`, techLead `claude-4-8-opus`, researcher `claude-4-8-opus`, validator `claude-4-8-opus`.


---
_Filed by an automated security-scan harness; AI-generated — review before acting._ Finding #879.