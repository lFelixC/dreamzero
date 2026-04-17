#!/usr/bin/env python3
"""Generate a lightweight HTML report for DreamZero RTC debug traces."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from html import escape
from pathlib import Path


HTML_BG = "#f7f4ec"
HTML_PANEL = "#fffdf7"
HTML_GRID = "#ddd5c6"
HTML_TEXT = "#2b2620"
HTML_MUTED = "#6f6659"
COLOR_JERK = "#0f6cbd"
COLOR_GAP = "#c77d00"
COLOR_TIMING = "#2f855a"
COLOR_DELAY = "#7b2cbf"
COLOR_ACTIVATE = "#d62828"
COLOR_WAIT = "#6c757d"
COLOR_CURRENT = "#2f855a"
COLOR_RAW = "#d62828"
COLOR_EXEC = "#0f6cbd"


@dataclass
class StepRecord:
    step: int
    chunk_id: int | None
    chunk_local_step_idx: int
    chunk_size: int
    activated_chunk: bool
    chunk_activation_source: str
    queried_policy: bool
    launched_prefetch: bool
    waited_for_policy: bool
    handoff_smoothed: bool
    request_delay_steps: int
    next_delay_steps: int
    policy_roundtrip_ms: float
    policy_wait_ms: float
    infer_total_ms: float
    joint_position: list[float]
    raw_action: list[float]
    executed_action: list[float]
    session_id: str
    instruction: str
    loop_step: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-path", required=True, help="Path to the jsonl trace written by main_dreamzero.py")
    parser.add_argument(
        "--output-html",
        default=None,
        help="Output html path. Defaults to <trace_stem>_<session_id>.html beside the trace.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session_id to visualize. Defaults to the last session found in the trace.",
    )
    parser.add_argument(
        "--detail-count",
        type=int,
        default=3,
        help="How many suspicious-step detail panels to include.",
    )
    parser.add_argument(
        "--detail-window",
        type=int,
        default=4,
        help="Number of steps to show before/after each suspicious step in detail panels.",
    )
    return parser.parse_args()


def _safe_float_list(values: object, expected: int) -> list[float]:
    if not isinstance(values, list):
        return [0.0] * expected
    floats = [float(v) for v in values[:expected]]
    if len(floats) < expected:
        floats.extend([0.0] * (expected - len(floats)))
    return floats


def load_records(trace_path: Path) -> list[dict]:
    records: list[dict] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse json on line {line_number}: {exc}") from exc
            if payload.get("trace_type") != "dreamzero_rtc_step":
                continue
            records.append(payload)
    if not records:
        raise ValueError(f"No rtc debug trace records found in {trace_path}")
    return records


def select_session(records: list[dict], requested_session_id: str | None) -> tuple[str, list[dict]]:
    session_order: list[str] = []
    by_session: dict[str, list[dict]] = {}
    for record in records:
        session_id = str(record.get("session_id") or record.get("episode_session_id") or "__unknown__")
        if session_id not in by_session:
            by_session[session_id] = []
            session_order.append(session_id)
        by_session[session_id].append(record)

    if requested_session_id is None:
        session_id = session_order[-1]
    else:
        session_id = requested_session_id
        if session_id not in by_session:
            known = ", ".join(session_order)
            raise ValueError(f"Session '{session_id}' not found. Known sessions: {known}")
    return session_id, by_session[session_id]


def normalize_records(raw_records: list[dict]) -> list[StepRecord]:
    normalized: list[StepRecord] = []
    sorted_records = sorted(
        raw_records,
        key=lambda item: (
            int(item.get("executed_step_idx", item.get("loop_step", 0))),
            int(item.get("loop_step", item.get("executed_step_idx", 0))),
        ),
    )
    for item in sorted_records:
        normalized.append(
            StepRecord(
                step=int(item.get("executed_step_idx", item.get("loop_step", 0))),
                chunk_id=None if item.get("current_chunk_id") is None else int(item["current_chunk_id"]),
                chunk_local_step_idx=int(item.get("chunk_local_step_idx", 0)),
                chunk_size=int(item.get("chunk_size", 0)),
                activated_chunk=bool(item.get("activated_chunk_this_step", False)),
                chunk_activation_source=str(item.get("chunk_activation_source", "none")),
                queried_policy=bool(item.get("queried_policy", False)),
                launched_prefetch=bool(item.get("launched_prefetch", False)),
                waited_for_policy=bool(item.get("waited_for_policy", False)),
                handoff_smoothed=bool(item.get("handoff_smoothed", False)),
                request_delay_steps=int(item.get("rtc_request_delay_steps", 0)),
                next_delay_steps=int(item.get("rtc_next_delay_steps", 0)),
                policy_roundtrip_ms=float(item.get("policy_roundtrip_ms", 0.0)),
                policy_wait_ms=float(item.get("policy_wait_ms", 0.0)),
                infer_total_ms=float(item.get("infer_total_ms", item.get("client_infer_ms", 0.0))),
                joint_position=_safe_float_list(item.get("joint_position", []), 7),
                raw_action=_safe_float_list(item.get("raw_action_before_handoff", []), 8),
                executed_action=_safe_float_list(item.get("executed_action", []), 8),
                session_id=str(item.get("session_id", "")),
                instruction=str(item.get("instruction", "")),
                loop_step=int(item.get("loop_step", item.get("executed_step_idx", 0))),
            )
        )
    return normalized


def max_abs_delta(lhs: list[float], rhs: list[float], dims: int = 7) -> float:
    return max((abs(float(a) - float(b)) for a, b in zip(lhs[:dims], rhs[:dims])), default=0.0)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = (len(ordered) - 1) * q
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(ordered[lo])
    frac = idx - lo
    return float(ordered[lo] * (1 - frac) + ordered[hi] * frac)


def build_svg_chart(
    x_values: list[int],
    series: list[tuple[str, str, list[float]]],
    vertical_markers: list[tuple[int, str, str]],
    width: int = 980,
    height: int = 280,
) -> str:
    if not x_values:
        return "<svg></svg>"

    left = 64
    right = 24
    top = 24
    bottom = 44
    chart_width = max(width - left - right, 1)
    chart_height = max(height - top - bottom, 1)

    all_y = [value for _, _, values in series for value in values]
    y_max = max(all_y) if all_y else 1.0
    y_min = min(all_y) if all_y else 0.0
    if y_max == y_min:
        y_max = y_min + 1.0
    y_pad = (y_max - y_min) * 0.08
    y_min -= y_pad
    y_max += y_pad

    x_min = min(x_values)
    x_max = max(x_values)
    if x_max == x_min:
        x_max = x_min + 1

    def map_x(x: float) -> float:
        return left + (float(x) - x_min) * chart_width / (x_max - x_min)

    def map_y(y: float) -> float:
        return top + chart_height - (float(y) - y_min) * chart_height / (y_max - y_min)

    grid_lines = []
    for tick_idx in range(5):
        y_value = y_min + (y_max - y_min) * tick_idx / 4
        y = map_y(y_value)
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + chart_width}" y2="{y:.2f}" stroke="{HTML_GRID}" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="12" fill="{HTML_MUTED}">{y_value:.3f}</text>'
        )

    x_tick_count = min(8, len(x_values))
    for tick_idx in range(x_tick_count):
        x_value = x_min + (x_max - x_min) * tick_idx / max(x_tick_count - 1, 1)
        x = map_x(x_value)
        grid_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + chart_height}" stroke="{HTML_GRID}" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{x:.2f}" y="{top + chart_height + 22}" text-anchor="middle" font-size="12" fill="{HTML_MUTED}">{int(round(x_value))}</text>'
        )

    marker_lines = []
    for step, color, label in vertical_markers:
        x = map_x(step)
        marker_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + chart_height}" stroke="{color}" stroke-width="1.5" stroke-dasharray="4 3" />'
        )
        marker_lines.append(
            f'<text x="{x + 3:.2f}" y="{top + 14}" font-size="11" fill="{color}">{escape(label)}</text>'
        )

    polylines = []
    legend_items = []
    legend_x = left
    legend_y = height - 12
    for idx, (label, color, values) in enumerate(series):
        points = " ".join(f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in zip(x_values, values))
        polylines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" />'
        )
        legend_offset = idx * 180
        legend_items.append(
            f'<line x1="{legend_x + legend_offset}" y1="{legend_y}" x2="{legend_x + legend_offset + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3" />'
            f'<text x="{legend_x + legend_offset + 32}" y="{legend_y + 4}" font-size="12" fill="{HTML_TEXT}">{escape(label)}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="auto" role="img">'
        f'{"".join(grid_lines)}'
        f'{"".join(marker_lines)}'
        f'{"".join(polylines)}'
        f'{"".join(legend_items)}'
        f'</svg>'
    )


def build_top_steps_table(top_steps: list[dict]) -> str:
    rows = []
    for item in top_steps:
        rows.append(
            "<tr>"
            f"<td>{item['step']}</td>"
            f"<td>{item['score']:.4f}</td>"
            f"<td>{item['action_step_delta_max']:.4f}</td>"
            f"<td>{item['action_obs_delta_max']:.4f}</td>"
            f"<td>{'yes' if item['activated_chunk'] else 'no'}</td>"
            f"<td>{escape(item['chunk_activation_source'])}</td>"
            f"<td>{item['request_delay_steps']}</td>"
            f"<td>{item['policy_roundtrip_ms']:.1f}</td>"
            f"<td>{'yes' if item['handoff_smoothed'] else 'no'}</td>"
            "</tr>"
        )
    return (
        "<table>"
        "<thead><tr>"
        "<th>step</th><th>score</th><th>action step delta</th><th>action-observation gap</th>"
        "<th>chunk activate</th><th>source</th><th>delay</th><th>roundtrip ms</th><th>smoothed</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def build_detail_panel(records: list[StepRecord], center_index: int, window: int) -> str:
    start = max(center_index - window, 0)
    end = min(center_index + window + 1, len(records))
    center = records[center_index]
    prev_action = records[center_index - 1].executed_action if center_index > 0 else center.executed_action
    jerk_by_joint = [
        abs(center.executed_action[j] - prev_action[j]) for j in range(7)
    ]
    joint_idx = max(range(7), key=lambda j: jerk_by_joint[j])
    panel_records = records[start:end]
    x_values = [item.step for item in panel_records]
    series = [
        ("current joint", COLOR_CURRENT, [item.joint_position[joint_idx] for item in panel_records]),
        ("raw action", COLOR_RAW, [item.raw_action[joint_idx] for item in panel_records]),
        ("executed action", COLOR_EXEC, [item.executed_action[joint_idx] for item in panel_records]),
    ]
    markers = [(center.step, COLOR_ACTIVATE, f"step {center.step}")]
    svg = build_svg_chart(x_values, series, markers, width=980, height=220)
    summary = (
        f"step={center.step}, joint={joint_idx}, chunk_id={center.chunk_id}, "
        f"activate={'yes' if center.activated_chunk else 'no'}, "
        f"source={center.chunk_activation_source}, "
        f"delay={center.request_delay_steps}, "
        f"roundtrip={center.policy_roundtrip_ms:.1f}ms, "
        f"smooth={'yes' if center.handoff_smoothed else 'no'}"
    )
    return (
        '<section class="panel">'
        f"<h3>Detail Around Suspicious Step {center.step}</h3>"
        f"<p class=\"muted\">{escape(summary)}</p>"
        f"{svg}"
        "</section>"
    )


def diagnose(records: list[StepRecord], top_steps: list[dict], action_step_deltas: list[float]) -> str:
    activation_deltas = [
        delta for record, delta in zip(records, action_step_deltas) if record.activated_chunk
    ]
    steady_deltas = [
        delta for record, delta in zip(records, action_step_deltas) if not record.activated_chunk
    ]
    top_activation_ratio = (
        sum(1 for item in top_steps if item["activated_chunk"]) / len(top_steps) if top_steps else 0.0
    )
    mean_activation = mean(activation_deltas)
    mean_steady = mean(steady_deltas)

    if activation_deltas and mean_activation > max(mean_steady * 1.5, mean_steady + 0.02) and top_activation_ratio >= 0.5:
        return (
            "高峰明显集中在 chunk 激活点，抽动更像是 RTC handoff / delay 对齐问题，"
            "优先检查 `rtc_inference_delay_steps`、`rtc_use_measured_delay` 和 `rtc_handoff_joint_blend`。"
        )
    if top_steps and sum(1 for item in top_steps if item["waited_for_policy"]) / len(top_steps) >= 0.5:
        return (
            "可疑步里等待 server 返回的比例很高，说明抽动可能和 prefetch 没赶上、"
            "实际延迟波动或 server roundtrip 抖动有关。"
        )
    return (
        "高峰没有强烈绑定在 chunk handoff 上，更像是模型本身输出不连续、"
        "观测状态漂移，或 relative/absolute action 对齐问题。"
    )


def build_report(records: list[StepRecord], session_id: str, detail_count: int, detail_window: int) -> str:
    x_values = [record.step for record in records]
    action_step_deltas: list[float] = []
    action_obs_deltas: list[float] = []
    raw_obs_deltas: list[float] = []
    for index, record in enumerate(records):
        prev_action = records[index - 1].executed_action if index > 0 else record.executed_action
        action_step_deltas.append(max_abs_delta(record.executed_action, prev_action))
        action_obs_deltas.append(max_abs_delta(record.executed_action, record.joint_position))
        raw_obs_deltas.append(max_abs_delta(record.raw_action, record.joint_position))

    suspicious_steps: list[dict] = []
    for index, record in enumerate(records):
        score = action_step_deltas[index] + 0.7 * action_obs_deltas[index]
        suspicious_steps.append(
            {
                "index": index,
                "step": record.step,
                "score": score,
                "action_step_delta_max": action_step_deltas[index],
                "action_obs_delta_max": action_obs_deltas[index],
                "raw_obs_delta_max": raw_obs_deltas[index],
                "activated_chunk": record.activated_chunk,
                "chunk_activation_source": record.chunk_activation_source,
                "request_delay_steps": record.request_delay_steps,
                "policy_roundtrip_ms": record.policy_roundtrip_ms,
                "waited_for_policy": record.waited_for_policy,
                "handoff_smoothed": record.handoff_smoothed,
            }
        )
    top_steps = sorted(suspicious_steps, key=lambda item: item["score"], reverse=True)[:10]

    markers = [(record.step, COLOR_ACTIVATE, "activate") for record in records if record.activated_chunk]
    markers.extend((record.step, COLOR_WAIT, "wait") for record in records if record.waited_for_policy)
    overview_svg = build_svg_chart(
        x_values=x_values,
        series=[
            ("action step delta max", COLOR_JERK, action_step_deltas),
            ("action-observation gap max", COLOR_GAP, action_obs_deltas),
        ],
        vertical_markers=markers,
    )
    timing_svg = build_svg_chart(
        x_values=x_values,
        series=[
            ("policy roundtrip ms", COLOR_TIMING, [record.policy_roundtrip_ms for record in records]),
            ("delay steps", COLOR_DELAY, [float(record.request_delay_steps) for record in records]),
        ],
        vertical_markers=[(record.step, COLOR_ACTIVATE, "activate") for record in records if record.activated_chunk],
    )

    activation_deltas = [delta for record, delta in zip(records, action_step_deltas) if record.activated_chunk]
    steady_deltas = [delta for record, delta in zip(records, action_step_deltas) if not record.activated_chunk]
    diagnosis = diagnose(records, top_steps, action_step_deltas)
    instruction = next((record.instruction for record in records if record.instruction), "")

    detail_panels = []
    for item in top_steps[: max(detail_count, 0)]:
        detail_panels.append(build_detail_panel(records, item["index"], detail_window))

    summary_items = [
        f"Session: <code>{escape(session_id)}</code>",
        f"Instruction: {escape(instruction or '(empty)')}",
        f"Total steps: {len(records)}",
        f"Chunk activations: {sum(1 for record in records if record.activated_chunk)}",
        f"Median action-step delta max: {median(action_step_deltas):.4f}",
        f"P95 action-step delta max: {percentile(action_step_deltas, 0.95):.4f}",
        f"Mean action-step delta on activations: {mean(activation_deltas):.4f}",
        f"Mean action-step delta off activations: {mean(steady_deltas):.4f}",
        f"Max policy roundtrip: {max((record.policy_roundtrip_ms for record in records), default=0.0):.1f} ms",
    ]

    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DreamZero RTC Trace Report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: {HTML_BG};
      --panel: {HTML_PANEL};
      --grid: {HTML_GRID};
      --text: {HTML_TEXT};
      --muted: {HTML_MUTED};
    }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
      background: linear-gradient(180deg, #efe8d8 0%, var(--bg) 18%, #f3eee4 100%);
      color: var(--text);
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px;
      font-weight: 600;
      letter-spacing: 0.01em;
    }}
    p, li {{
      line-height: 1.5;
      margin: 0;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
    }}
    .muted {{
      color: var(--muted);
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid #e7dfd0;
      border-radius: 18px;
      padding: 18px 18px 16px;
      box-shadow: 0 8px 24px rgba(89, 70, 39, 0.06);
      margin-top: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 8px 10px;
      border-bottom: 1px solid #ece3d4;
      vertical-align: top;
    }}
    th {{
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 0.9em;
      background: #f4efe3;
      padding: 0.1em 0.35em;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>DreamZero RTC Trace Report</h1>
      <p class="muted">{escape(diagnosis)}</p>
      <ul>{"".join(f"<li>{item}</li>" for item in summary_items)}</ul>
    </section>

    <section class="panel">
      <h2>Overview</h2>
      <p class="muted">蓝线是相邻执行动作的最大关节跳变，橙线是当前观测关节到执行动作的最大偏差；红虚线表示 chunk 激活，灰虚线表示这一步确实等了 policy 返回。</p>
      {overview_svg}
    </section>

    <section class="panel">
      <h2>Timing And Delay</h2>
      <p class="muted">这里可以看 jerk 是否和 roundtrip 抖动、delay step 变化一起出现。</p>
      {timing_svg}
    </section>

    <section class="panel">
      <h2>Top Suspicious Steps</h2>
      <p class="muted">score = action-step-delta + 0.7 * action-observation-gap。</p>
      {build_top_steps_table(top_steps)}
    </section>

    {"".join(detail_panels)}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    trace_path = Path(args.trace_path).expanduser().resolve()
    records = load_records(trace_path)
    session_id, session_records = select_session(records, args.session_id)
    normalized = normalize_records(session_records)
    report_html = build_report(normalized, session_id, args.detail_count, args.detail_window)

    if args.output_html is None:
        safe_session = session_id.replace("/", "_")
        output_html = trace_path.with_name(f"{trace_path.stem}_{safe_session}.html")
    else:
        output_html = Path(args.output_html).expanduser().resolve()

    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(report_html, encoding="utf-8")
    print(f"Wrote RTC trace report to {output_html}")


if __name__ == "__main__":
    main()
