# Listing 10.3 — Trace execution graph builder: JSONL spans + messages to DOT.

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SpanNode:
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: str
    t_start: float
    t_end: float
    status: str


@dataclass
class MsgEdge:
    msg_id: str
    src: str
    dst: str
    msg_type: str
    t_send: float
    t_recv: float
    sender_span_id: str
    receiver_span_id: str


def load_spans(path: str, trace_id: str) -> Dict[str, SpanNode]:
    spans: Dict[str, SpanNode] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("trace_id") != trace_id:
                continue
            spans[r["span_id"]] = SpanNode(
                span_id=r["span_id"],
                parent_span_id=r.get("parent_span_id"),
                name=r.get("name", ""),
                kind=r.get("kind", "internal"),
                t_start=float(r.get("t_start", 0.0)),
                t_end=float(r.get("t_end", 0.0)),
                status=r.get("status", "ok"),
            )
    return spans


def load_message_rows(path: str, trace_id: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("trace_id") != trace_id:
                continue
            rows.append(r)
    return rows


def pair_send_recv(rows: List[Dict[str, Any]]) -> List[MsgEdge]:
    sends: Dict[str, Dict[str, Any]] = {}
    recvs: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        mid = r.get("msg_id")
        if not mid:
            continue
        if r.get("direction") == "send":
            sends[mid] = r
        elif r.get("direction") == "recv":
            # keep the first recv for this msg_id (single-hop model)
            if mid not in recvs:
                recvs[mid] = r

    edges: List[MsgEdge] = []
    for msg_id, s in sends.items():
        r = recvs.get(msg_id)
        if r is None:
            continue
        sid = s.get("emit_span_id") or ""
        rid = r.get("emit_span_id") or ""
        if not sid or not rid:
            continue
        edges.append(
            MsgEdge(
                msg_id=msg_id,
                src=s.get("src", ""),
                dst=s.get("dst", ""),
                msg_type=s.get("msg_type", ""),
                t_send=float(s.get("t", 0.0)),
                t_recv=float(r.get("t", 0.0)),
                sender_span_id=sid,
                receiver_span_id=rid,
            )
        )
    return edges


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def to_dot(spans: Dict[str, SpanNode], edges: List[MsgEdge]) -> str:
    def label(s: SpanNode) -> str:
        dur_ms = (s.t_end - s.t_start) * 1000.0
        return _esc(f"{s.name}\n{s.kind}\n{dur_ms:.1f} ms\n{s.status}")

    out: List[str] = []
    out.append("digraph trace {")
    out.append('  rankdir="LR";')
    out.append('  node [shape=box];')

    for sid, s in spans.items():
        out.append(f'  "{sid}" [label="{label(s)}"];')

    for sid, s in spans.items():
        if s.parent_span_id and s.parent_span_id in spans:
            out.append(f'  "{s.parent_span_id}" -> "{sid}" [style=dashed,label="parent"];')

    for e in edges:
        if e.sender_span_id in spans and e.receiver_span_id in spans:
            dt_ms = (e.t_recv - e.t_send) * 1000.0
            out.append(
                f'  "{e.sender_span_id}" -> "{e.receiver_span_id}" '
                f'[color=blue,label="{_esc(e.msg_type)}:{_esc(e.src)}->{_esc(e.dst)}\\n{dt_ms:.1f} ms"];'
            )

    out.append("}")
    return "\n".join(out)
