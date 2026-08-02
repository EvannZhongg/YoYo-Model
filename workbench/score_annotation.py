"""Browser-based score-event annotation workbench and metadata contract."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any

import gradio as gr

from common.files import atomic_write_text
from config import BASE_DIR


SCHEMA_VERSION = "yoyo_score_annotation_v2"
DIVISIONS = ("1A", "2A", "3A", "4A", "5A")
MAJOR_PENALTIES = {
    "restart": {"display_name": "重启", "score_delta": -1},
    "discard": {"display_name": "弃用", "score_delta": -3},
    "disassembly": {"display_name": "解体", "score_delta": -5},
}
ANCHOR_SOURCES = ("evidence_end_default", "manual")
EXCLUSION_REASONS = ("defocus", "occlusion", "corrupted_frames", "other")
SCENE_TYPES = ("irrelevant_scene", "player_entry_exit")
SCORE_ANNOTATION_DIR = BASE_DIR / "annotations" / "score_annotations"
SCORE_VIDEO_DIRS = (BASE_DIR / "videos",)
_STORAGE_LOCK = threading.Lock()


def validate_score_annotation(document: dict[str, Any]) -> None:
    """Validate the stable fields consumed by later score-model tooling."""
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {document.get('schema_version')!r}")
    competition = document.get("competition") or {}
    if competition.get("division") not in DIVISIONS:
        raise ValueError("competition.division must be one of 1A/2A/3A/4A/5A")
    judge = str((document.get("annotator") or {}).get("judge") or "").strip()
    if not judge:
        raise ValueError("annotator.judge is required")
    video = document.get("video") or {}
    source_path = str(video.get("source_path") or "").strip()
    if not source_path:
        raise ValueError("video.source_path is required")
    _managed_score_video_path(source_path)

    for index, event in enumerate(document.get("events") or []):
        label = event.get("label") or {}
        timing = event.get("timing") or {}
        family = label.get("family")
        delta = label.get("score_delta")
        if family == "positive" and (not isinstance(delta, int) or not 0 <= delta <= 10):
            raise ValueError(f"events[{index}] positive score must be between 0 and 10")
        if family == "negative" and (not isinstance(delta, int) or not -10 <= delta <= -1):
            raise ValueError(f"events[{index}] negative score must be between -10 and -1")
        if family == "major_penalty":
            penalty_type = label.get("penalty_type")
            expected = MAJOR_PENALTIES.get(str(penalty_type), {}).get("score_delta")
            if expected is None or delta != expected:
                raise ValueError(f"events[{index}] has an invalid major penalty")
        if family not in {"positive", "negative", "major_penalty"}:
            raise ValueError(f"events[{index}] has an invalid label family")
        start = timing.get("evidence_start_s")
        anchor = timing.get("anchor_s")
        end = timing.get("evidence_end_s")
        if not all(isinstance(value, (int, float)) for value in (start, anchor, end)):
            raise ValueError(f"events[{index}] timing values must be numeric")
        if start < 0 or not start <= anchor <= end:
            raise ValueError(f"events[{index}] must satisfy evidence_start <= anchor <= evidence_end")
        anchor_source = timing.get("anchor_source")
        if anchor_source not in ANCHOR_SOURCES:
            raise ValueError(f"events[{index}].timing.anchor_source must be evidence_end_default or manual")

    excluded_intervals = document.get("excluded_intervals", [])
    if not isinstance(excluded_intervals, list):
        raise ValueError("excluded_intervals must be a list")
    for index, interval in enumerate(excluded_intervals):
        if not isinstance(interval, dict):
            raise ValueError(f"excluded_intervals[{index}] must be an object")
        start = interval.get("start_s")
        end = interval.get("end_s")
        if not all(isinstance(value, (int, float)) for value in (start, end)):
            raise ValueError(f"excluded_intervals[{index}] timing values must be numeric")
        if start < 0 or start >= end:
            raise ValueError(f"excluded_intervals[{index}] must satisfy 0 <= start_s < end_s")
        if interval.get("reason") not in EXCLUSION_REASONS:
            raise ValueError(f"excluded_intervals[{index}].reason is invalid")
        if interval.get("training_eligible") is not False:
            raise ValueError(f"excluded_intervals[{index}].training_eligible must be false")

    scene_intervals = document.get("scene_intervals")
    if not isinstance(scene_intervals, list):
        raise ValueError("scene_intervals must be a list")
    for index, interval in enumerate(scene_intervals):
        if not isinstance(interval, dict):
            raise ValueError(f"scene_intervals[{index}] must be an object")
        start = interval.get("start_s")
        end = interval.get("end_s")
        if not all(isinstance(value, (int, float)) for value in (start, end)):
            raise ValueError(f"scene_intervals[{index}] timing values must be numeric")
        if start < 0 or start >= end:
            raise ValueError(f"scene_intervals[{index}] must satisfy 0 <= start_s < end_s")
        if interval.get("scene_type") not in SCENE_TYPES:
            raise ValueError(f"scene_intervals[{index}].scene_type is invalid")

    serve_receive_events = document.get("serve_receive_events", [])
    if not isinstance(serve_receive_events, list):
        raise ValueError("serve_receive_events must be a list")
    for index, event in enumerate(serve_receive_events):
        marker = event.get("marker") or {}
        timing = event.get("timing") or {}
        if marker.get("track") != "serve_receive" or marker.get("type") not in {"begin", "end"}:
            raise ValueError(f"serve_receive_events[{index}].marker is invalid")
        keyframe = timing.get("anchor_s")
        start = timing.get("evidence_start_s")
        end = timing.get("evidence_end_s")
        if not all(isinstance(value, (int, float)) for value in (keyframe, start, end)):
            raise ValueError(f"serve_receive_events[{index}] timing values must be numeric")
        if start < 0 or not start <= keyframe <= end:
            raise ValueError(f"serve_receive_events[{index}] has invalid timing")
        for field in ("anchor_frame_index", "evidence_start_frame_index", "evidence_end_frame_index"):
            if not isinstance(timing.get(field), int):
                raise ValueError(f"serve_receive_events[{index}].timing.{field} must be an integer")


def _storage_identity(browser_identity: str) -> str:
    return hashlib.sha256(browser_identity.encode("utf-8")).hexdigest()[:16]


def _score_annotation_path(document: dict[str, Any]) -> Path:
    video = document.get("video") or {}
    browser_identity = str(video.get("browser_identity") or "").strip()
    if not browser_identity:
        raise ValueError("video.browser_identity is required")
    stem = Path(str(video.get("file_name") or "video")).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "video"
    return SCORE_ANNOTATION_DIR / f"{safe_stem[:80]}_{_storage_identity(browser_identity)}.json"


def _read_score_document(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_score_annotation(document)
    return document


def save_score_annotation(document_json: str | dict[str, Any]) -> dict[str, Any]:
    """Atomically persist one browser annotation document to the repository."""
    document = json.loads(document_json) if isinstance(document_json, str) else document_json
    if not isinstance(document, dict):
        raise ValueError("score annotation must be a JSON object")
    validate_score_annotation(document)
    output_path = _score_annotation_path(document)
    payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    with _STORAGE_LOCK:
        atomic_write_text(output_path, payload)
    return {
        "storage_key": output_path.name,
        "path": str(output_path),
        "document": document,
    }


def load_score_annotation(browser_identity: str) -> dict[str, Any] | None:
    """Load the disk-backed session for a selected local video identity."""
    identity = str(browser_identity or "").strip()
    if not identity or not SCORE_ANNOTATION_DIR.is_dir():
        return None
    matches = sorted(SCORE_ANNOTATION_DIR.glob(f"*_{_storage_identity(identity)}.json"))
    if not matches:
        return None
    path = matches[-1]
    return {"storage_key": path.name, "path": str(path), "document": _read_score_document(path)}


def _managed_score_annotation_path(storage_key: str) -> Path:
    name = Path(str(storage_key or "")).name
    if name != storage_key or not name.endswith(".json"):
        raise ValueError("invalid score annotation storage key")
    return SCORE_ANNOTATION_DIR / name


def _managed_score_video_path(source_path: str) -> Path:
    relative_path = Path(str(source_path or "").strip())
    if not str(relative_path) or relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("invalid video.source_path")
    path = (BASE_DIR / relative_path).resolve()
    allowed_roots = [root.resolve() for root in SCORE_VIDEO_DIRS]
    if not any(path == root or path.is_relative_to(root) for root in allowed_roots):
        raise ValueError("video.source_path must be inside a managed video directory")
    return path


def resolve_score_video_source(video_metadata: dict[str, Any]) -> str:
    """Resolve an imported browser File to the source_path stored in a new session."""
    file_name = str(video_metadata.get("file_name") or "").strip()
    expected_size = video_metadata.get("file_size_bytes")
    expected_modified_ms = video_metadata.get("last_modified_ms")
    if not file_name or not isinstance(expected_size, int) or not isinstance(expected_modified_ms, int):
        raise ValueError("video file metadata is incomplete")
    matches: list[Path] = []
    for root in SCORE_VIDEO_DIRS:
        if not root.is_dir():
            continue
        for candidate in root.rglob(file_name):
            if not candidate.is_file():
                continue
            stat = candidate.stat()
            if stat.st_size == expected_size and int(stat.st_mtime * 1000) == expected_modified_ms:
                matches.append(candidate.resolve())
    if len(matches) != 1:
        raise ValueError("视频必须唯一匹配 workbench 的 videos 目录")
    return matches[0].relative_to(BASE_DIR.resolve()).as_posix()


def load_score_annotation_session(storage_key: str) -> dict[str, Any]:
    """Load a managed session and resolve its exact source video without a file picker."""
    path = _managed_score_annotation_path(storage_key)
    if not path.is_file():
        raise ValueError("score annotation session does not exist")
    document = _read_score_document(path)
    video_path = _managed_score_video_path(document["video"]["source_path"])
    if not video_path.is_file():
        raise ValueError("计分会话对应的视频文件不存在")
    result = {
        "storage_key": path.name,
        "path": str(path),
        "document": document,
        "video_path": str(video_path),
    }
    gr.set_static_paths(paths=[video_path])
    return result


def list_score_annotations(_component_placeholder: object = None) -> list[dict[str, Any]]:
    """List valid saved sessions, newest first, for the management drawer."""
    if not SCORE_ANNOTATION_DIR.is_dir():
        return []
    sessions: list[dict[str, Any]] = []
    for path in SCORE_ANNOTATION_DIR.glob("*.json"):
        try:
            document = _read_score_document(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        sessions.append({"storage_key": path.name, "path": str(path), "document": document})
    sessions.sort(key=lambda item: str(item["document"].get("updated_at") or ""), reverse=True)
    return sessions


def delete_score_annotation(storage_key: str) -> bool:
    """Delete one managed session without accepting arbitrary filesystem paths."""
    path = _managed_score_annotation_path(storage_key)
    with _STORAGE_LOCK:
        if not path.is_file():
            return False
        path.unlink()
    return True


SCORE_ANNOTATION_HTML = r"""
<div class="ysa" data-yoyo-score-annotation>
  <header class="ysa__header">
    <div>
      <h2>悠悠球计分标注</h2>
      <p id="ysa-session-state">选择本地视频开始新的标注会话</p>
    </div>
    <div class="ysa__header-actions">
      <button class="ysa__button" id="ysa-manager-open" type="button">会话管理 <span id="ysa-manager-count">0</span></button>
      <label class="ysa__button ysa__button--primary" for="ysa-video-file">导入视频</label>
      <input id="ysa-video-file" type="file" accept="video/*" hidden>
      <label class="ysa__button" for="ysa-metadata-file">导入元数据</label>
      <input id="ysa-metadata-file" type="file" accept="application/json,.json" hidden>
      <button class="ysa__button" id="ysa-export" type="button" disabled>导出 JSON</button>
    </div>
  </header>

  <section class="ysa__session" aria-label="标注会话">
    <label>组别
      <select id="ysa-division">
        <option>1A</option><option>2A</option><option>3A</option><option>4A</option><option>5A</option>
      </select>
    </label>
    <label>裁判
      <input id="ysa-judge" type="text" value="judge1" maxlength="80" autocomplete="off">
    </label>
    <label>帧率
      <input id="ysa-fps" type="number" value="30" min="1" max="240" step="0.001">
    </label>
    <div class="ysa__identity" id="ysa-video-identity">尚未选择视频</div>
  </section>

  <main class="ysa__workspace">
    <section class="ysa__player-panel" aria-label="视频播放器">
      <div class="ysa__viewport" id="ysa-viewport">
        <div class="ysa__empty" id="ysa-empty">
          <strong>导入需要标注的本地视频</strong>
          <span>视频仅在当前浏览器中读取</span>
        </div>
        <video id="ysa-video" preload="metadata" controls playsinline></video>
      </div>
      <div class="ysa__transport">
        <button type="button" id="ysa-prev" title="上一帧" disabled>上一帧</button>
        <button type="button" id="ysa-next" title="下一帧" disabled>下一帧</button>
        <output id="ysa-timecode">00:00.000 · f0</output>
        <label class="ysa__zoom">缩放
          <input id="ysa-zoom" type="range" min="100" max="300" value="100" step="25" disabled>
          <output id="ysa-zoom-value">100%</output>
        </label>
      </div>
    </section>

    <aside class="ysa__editor" aria-label="计分事件编辑器">
      <div class="ysa__editor-title">
        <h3 id="ysa-editor-title">新建计分事件</h3>
        <button type="button" id="ysa-new" class="ysa__button ysa__button--quiet" disabled>清空</button>
      </div>
      <fieldset id="ysa-event-fields" disabled>
        <legend>计分事件标签</legend>
        <div class="ysa__segmented" id="ysa-family">
          <button type="button" data-family="positive" class="is-active">正向计分</button>
          <button type="button" data-family="negative">负向计分</button>
          <button type="button" data-family="major_penalty">重点扣分</button>
        </div>
        <label id="ysa-score-row">分值
          <input id="ysa-score" type="number" min="0" max="10" value="1" step="1">
        </label>
        <label id="ysa-penalty-row" hidden>扣分类别
          <select id="ysa-penalty">
            <option value="restart">重启（-1）</option>
            <option value="discard">弃用（-3）</option>
            <option value="disassembly">解体（-5）</option>
          </select>
        </label>

        <div class="ysa__timing-heading">
          <span>时间范围</span>
          <button type="button" id="ysa-use-anchor">锚点取当前帧</button>
        </div>
        <label>Anchor
          <div class="ysa__time-input"><input id="ysa-anchor" type="number" min="0" step="0.001" value="0"><button type="button" data-set-time="anchor">取当前</button></div>
        </label>
        <div class="ysa__time-grid">
          <label>Evidence 起点
            <div class="ysa__time-input"><input id="ysa-start" type="number" min="0" step="0.001" value="0"><button type="button" data-set-time="start">取当前</button></div>
          </label>
          <label>Evidence 终点
            <div class="ysa__time-input"><input id="ysa-end" type="number" min="0" step="0.001" value="0"><button type="button" data-set-time="end">取当前</button></div>
          </label>
        </div>
        <label>动作名称 <span class="ysa__optional">可选</span>
          <input id="ysa-action" type="text" maxlength="160" placeholder="不参与必需类别训练" autocomplete="off">
        </label>
        <div class="ysa__validation" id="ysa-validation" role="status"></div>
        <div class="ysa__editor-actions">
          <button type="button" class="ysa__button ysa__button--primary" id="ysa-save">记录起点</button>
          <button type="button" class="ysa__button ysa__button--danger" id="ysa-delete" hidden>删除</button>
        </div>
      </fieldset>
    </aside>
  </main>

  <section class="ysa__timeline-panel" aria-label="计分、场景与不可标记区间时间轴">
    <div class="ysa__timeline-heading">
      <h3>分轨时间轴</h3>
      <div class="ysa__timeline-tools">
        <span class="ysa__serve-receive-actions">发球/收球
          <button type="button" class="ysa__button" id="ysa-serve-begin" disabled>begin</button>
          <button type="button" class="ysa__button" id="ysa-serve-end" disabled>end</button>
        </span>
        <label>场景标签
          <select id="ysa-scene-type" title="无关场景包含广告、赞助商页等无需细分的内容" disabled>
            <option value="irrelevant_scene">无关场景</option>
            <option value="player_entry_exit">选手入/离场</option>
          </select>
        </label>
        <button type="button" class="ysa__button ysa__button--scene" id="ysa-scene-toggle" disabled>标记场景起点</button>
        <label>不可标记原因
          <select id="ysa-exclusion-reason" disabled>
            <option value="defocus">画面虚焦</option>
            <option value="occlusion">主体遮挡</option>
            <option value="corrupted_frames">画面损坏</option>
            <option value="other">其他</option>
          </select>
        </label>
        <button type="button" class="ysa__button ysa__button--exclude" id="ysa-exclusion-toggle" disabled>标记不可用起点</button>
        <label>时间轴缩放
          <input id="ysa-timeline-zoom" type="range" min="1" max="8" value="1" step="0.5" disabled>
          <output id="ysa-timeline-zoom-value">1×</output>
        </label>
      </div>
    </div>
    <div class="ysa__timeline-scroll" id="ysa-timeline-scroll">
      <div class="ysa__timeline-content" id="ysa-timeline-content">
        <div class="ysa__ruler"><span class="ysa__track-label">时间</span><div class="ysa__ruler-lane" id="ysa-ruler-lane"></div></div>
        <div class="ysa__track-row" data-family="positive"><span class="ysa__track-label">正向计分</span><div class="ysa__track-lane" data-track="positive" role="group" aria-label="正向计分轨"><div class="ysa__track-playhead" role="slider" tabindex="0" aria-label="播放位置" aria-valuemin="0" title="拖动定位播放帧"></div></div></div>
        <div class="ysa__track-row" data-family="negative"><span class="ysa__track-label">负向计分</span><div class="ysa__track-lane" data-track="negative" role="group" aria-label="负向计分轨"><div class="ysa__track-playhead" role="slider" tabindex="0" aria-label="播放位置" aria-valuemin="0" title="拖动定位播放帧"></div></div></div>
        <div class="ysa__track-row" data-family="major_penalty"><span class="ysa__track-label">重点扣分</span><div class="ysa__track-lane" data-track="major_penalty" role="group" aria-label="重点扣分轨"><div class="ysa__track-playhead" role="slider" tabindex="0" aria-label="播放位置" aria-valuemin="0" title="拖动定位播放帧"></div></div></div>
        <div class="ysa__track-row ysa__track-row--scene" data-family="scene"><span class="ysa__track-label">场景标注</span><div class="ysa__track-lane" data-track="scene" role="group" aria-label="无关场景与选手入离场标注轨"><div class="ysa__track-playhead" role="slider" tabindex="0" aria-label="播放位置" aria-valuemin="0" title="拖动定位播放帧"></div></div></div>
        <div class="ysa__track-row ysa__track-row--serve-receive" data-family="serve_receive"><span class="ysa__track-label">发球/收球</span><div class="ysa__track-lane" data-track="serve_receive" role="group" aria-label="发球和收球标注轨"><div class="ysa__track-playhead" role="slider" tabindex="0" aria-label="播放位置" aria-valuemin="0" title="拖动定位播放帧"></div></div></div>
        <div class="ysa__track-row ysa__track-row--excluded" data-family="excluded"><span class="ysa__track-label">不可标记</span><div class="ysa__track-lane" data-track="excluded" role="group" aria-label="不可标记区间轨"><div class="ysa__track-playhead" role="slider" tabindex="0" aria-label="播放位置" aria-valuemin="0" title="拖动定位播放帧"></div></div></div>
      </div>
    </div>
  </section>

  <section class="ysa__events" aria-label="已标注事件">
    <div class="ysa__events-heading">
      <h3>已标注事件 <span id="ysa-event-count">0</span></h3>
      <div><span>总分</span><strong id="ysa-total-score">0</strong></div>
    </div>
    <div class="ysa__table-wrap">
      <table>
        <thead><tr><th>#</th><th>标签</th><th>分值</th><th>Anchor</th><th>Evidence interval</th><th>动作名称</th><th>更新</th></tr></thead>
        <tbody id="ysa-event-list"><tr><td colspan="7" class="ysa__no-events">尚无计分事件</td></tr></tbody>
      </table>
    </div>
  </section>
  <div class="ysa__drawer-backdrop" id="ysa-manager-backdrop"></div>
  <aside class="ysa__drawer" id="ysa-manager" aria-label="计分标注会话管理" aria-hidden="true">
    <header class="ysa__drawer-header">
      <div><h3>会话管理</h3><span>本地 JSON 文件</span></div>
      <button type="button" id="ysa-manager-close" aria-label="关闭会话管理" title="关闭">×</button>
    </header>
    <div class="ysa__storage-location"><span>默认路径</span><code>annotations/score_annotations</code></div>
    <div class="ysa__session-list" id="ysa-session-list"></div>
  </aside>
  <div class="ysa__toast" id="ysa-toast" role="status" aria-live="polite"></div>
</div>
"""


SCORE_ANNOTATION_CSS = r"""
.ysa { --ink:#202321; --muted:#696f6a; --line:#d9ddd8; --surface:#fff; --soft:#f4f6f3; --accent:#176b55; --accent-soft:#e5f3ed; --danger:#b23a3a; background:var(--surface); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,sans-serif; width:100%; }
.ysa * { box-sizing:border-box; }
.ysa [hidden] { display:none!important; }
.ysa button,.ysa input,.ysa select { font:inherit; letter-spacing:0; }
.ysa__header { align-items:center; border-bottom:1px solid var(--line); display:flex; gap:20px; justify-content:space-between; padding:18px 4px 16px; }
.ysa h2,.ysa h3,.ysa p { color:var(--ink); margin:0; }
.ysa h2 { font-size:22px; font-weight:700; letter-spacing:0; }
.ysa h3 { font-size:15px; font-weight:700; letter-spacing:0; }
.ysa__header p { color:var(--muted); font-size:13px; margin-top:4px; }
.ysa__header-actions,.ysa__editor-actions { display:flex; flex-wrap:wrap; gap:8px; }
.ysa__header-actions #ysa-manager-count { background:#edf0ed; border-radius:9px; color:#566057; font-size:10px; line-height:16px; min-width:16px; padding:0 5px; text-align:center; }
.ysa__button,.ysa__transport button,.ysa__time-input button,.ysa__timing-heading button { align-items:center; background:#fff; border:1px solid #c9cec9; border-radius:6px; color:#303531; cursor:pointer; display:inline-flex; font-size:13px; font-weight:600; justify-content:center; min-height:34px; padding:7px 12px; }
.ysa__button:hover,.ysa__transport button:hover,.ysa__time-input button:hover,.ysa__timing-heading button:hover { background:var(--soft); }
.ysa button:disabled,.ysa__button[disabled] { cursor:not-allowed; opacity:.42; }
.ysa__button--primary { background:var(--accent); border-color:var(--accent); color:#fff; }
.ysa__button--primary:hover { background:#105a47; }
.ysa__button--quiet { min-height:30px; padding:5px 9px; }
.ysa__button--danger { border-color:#e3baba; color:var(--danger); }
.ysa__session { align-items:end; background:var(--soft); border-bottom:1px solid var(--line); display:grid; gap:12px; grid-template-columns:120px minmax(160px,220px) 110px 1fr; padding:12px 14px; }
.ysa label { color:#515651; display:grid; font-size:12px; font-weight:650; gap:5px; }
.ysa input,.ysa select { background:#fff; border:1px solid #cbd0cb; border-radius:5px; color:var(--ink); height:36px; min-width:0; padding:6px 9px; width:100%; }
.ysa input:focus,.ysa select:focus,.ysa button:focus-visible,.ysa__button:focus-visible { border-color:var(--accent); box-shadow:0 0 0 2px rgba(23,107,85,.15); outline:none; }
.ysa__identity { align-items:center; color:var(--muted); display:flex; font-size:12px; min-height:36px; overflow:hidden; padding:0 4px; text-overflow:ellipsis; white-space:nowrap; }
.ysa__workspace { display:grid; grid-template-columns:minmax(0,1.65fr) minmax(320px,.75fr); min-height:560px; }
.ysa__player-panel { background:#202320; border-right:1px solid var(--line); display:grid; grid-template-rows:minmax(400px,1fr) auto; min-width:0; }
.ysa__viewport { align-items:center; display:flex; justify-content:center; min-height:400px; overflow:auto; position:relative; }
.ysa__viewport video { display:none; height:auto; max-height:none; max-width:none; width:100%; }
.ysa__viewport video.is-ready { display:block; }
.ysa__empty { align-items:center; color:#d9ddd8; display:flex; flex-direction:column; gap:7px; justify-content:center; inset:0; position:absolute; }
.ysa__empty span { color:#9ba29c; font-size:13px; }
.ysa__transport { align-items:center; background:#2c302d; border-top:1px solid #414641; display:flex; gap:8px; min-height:52px; padding:8px 12px; }
.ysa__transport button { background:#393e3a; border-color:#535a54; color:#fff; min-width:74px; }
.ysa__transport output { color:#f3f5f3; font-family:"SFMono-Regular",Consolas,monospace; font-size:13px; margin-right:auto; white-space:nowrap; }
.ysa__zoom { align-items:center; color:#d5d9d5; display:flex; font-size:12px; gap:8px; grid-auto-flow:column; }
.ysa__zoom input { accent-color:#78bda8; height:auto; padding:0; width:110px; }
.ysa__zoom output { color:#d5d9d5; font-family:inherit; margin:0; min-width:38px; }
.ysa__editor { background:#fff; min-width:0; padding:18px; }
.ysa__editor-title,.ysa__events-heading { align-items:center; display:flex; justify-content:space-between; }
.ysa fieldset { border:0; margin:18px 0 0; padding:0; }
.ysa fieldset:disabled { opacity:.48; }
.ysa legend { font-size:12px; font-weight:700; margin-bottom:7px; }
.ysa__segmented { display:grid; grid-template-columns:1fr 1fr 1fr; margin-bottom:16px; }
.ysa__segmented button { background:#fff; border:1px solid #cbd0cb; border-radius:0; color:#4a504b; cursor:pointer; font-size:12px; font-weight:650; min-height:38px; padding:6px; }
.ysa__segmented button:first-child { border-radius:5px 0 0 5px; }
.ysa__segmented button:last-child { border-radius:0 5px 5px 0; }
.ysa__segmented button + button { border-left:0; }
.ysa__segmented button.is-active { background:var(--accent-soft); color:#0f5a45; }
.ysa__editor label { margin-bottom:13px; }
.ysa__timing-heading { align-items:center; border-top:1px solid var(--line); display:flex; font-size:12px; font-weight:700; justify-content:space-between; margin:18px 0 10px; padding-top:14px; }
.ysa__timing-heading button { min-height:29px; padding:4px 8px; }
.ysa__time-grid { display:grid; gap:9px; grid-template-columns:1fr 1fr; }
.ysa__time-input { display:grid; grid-template-columns:minmax(0,1fr) auto; }
.ysa__time-input input { border-radius:5px 0 0 5px; }
.ysa__time-input button { border-left:0; border-radius:0 5px 5px 0; font-size:11px; min-height:36px; padding:6px; }
.ysa__optional { color:#8a908b; font-weight:500; }
.ysa__validation { color:var(--danger); font-size:12px; min-height:28px; }
.ysa__editor-actions { border-top:1px solid var(--line); padding-top:14px; }
.ysa__editor-actions .ysa__button--primary { flex:1; }
.ysa__timeline-panel { background:#f7f8f6; border-top:1px solid var(--line); padding:12px 4px 4px; }
.ysa__timeline-heading { align-items:center; display:flex; gap:16px; justify-content:space-between; margin:0 8px 10px; }
.ysa__timeline-tools { align-items:center; display:flex; flex-wrap:wrap; gap:8px; justify-content:flex-end; }
.ysa__timeline-heading label { align-items:center; display:flex; flex-direction:row; gap:8px; margin:0; }
.ysa__timeline-heading select { height:34px; min-width:112px; width:auto; }
.ysa__timeline-heading input { accent-color:var(--accent); height:auto; padding:0; width:150px; }
.ysa__timeline-heading output { color:var(--muted); min-width:28px; }
.ysa__button--exclude { border-color:#b9a9d5; color:#60458b; }
.ysa__button--exclude.is-recording { background:#60458b; border-color:#60458b; color:#fff; }
.ysa__button--exclude.is-selected { border-color:var(--danger); color:var(--danger); }
.ysa__button--scene { border-color:#83b9c2; color:#276b76; }
.ysa__button--scene.is-recording { background:#276b76; border-color:#276b76; color:#fff; }
.ysa__button--scene.is-selected { border-color:var(--danger); color:var(--danger); }
.ysa__timeline-scroll { background:#292d2a; border:1px solid #3c423d; overflow-x:auto; overscroll-behavior-x:contain; }
.ysa__timeline-content { min-width:100%; width:100%; }
.ysa__ruler,.ysa__track-row { display:grid; grid-template-columns:96px minmax(0,1fr); min-width:0; }
.ysa__ruler { background:#242825; height:26px; }
.ysa__track-label { align-items:center; background:#333834; border-right:1px solid #4b514c; color:#dce0dc; display:flex; font-size:11px; font-weight:650; left:0; padding:0 10px; position:sticky; z-index:8; }
.ysa__ruler-lane { border-bottom:1px solid #454a46; min-width:0; position:relative; }
.ysa__tick { border-left:1px solid #5a605b; bottom:0; color:#b9beba; font-family:"SFMono-Regular",Consolas,monospace; font-size:9px; height:8px; padding-left:4px; position:absolute; white-space:nowrap; }
.ysa__track-row { border-bottom:1px solid #454a46; height:48px; }
.ysa__track-row:last-child { border-bottom:0; }
.ysa__track-lane { background-color:#2d312e; background-image:linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px); background-size:5% 100%; cursor:crosshair; min-width:0; position:relative; touch-action:none; }
.ysa__track-lane:hover { background-color:#313632; }
.ysa__track-playhead { background:transparent; bottom:0; cursor:col-resize; left:0; position:absolute; top:0; transform:translateX(-50%); width:15px; z-index:7; }
.ysa__track-playhead::before { background:#e29c35; bottom:0; content:""; left:6px; position:absolute; top:0; width:2px; }
.ysa__track-playhead:hover::before,.ysa__track-playhead.is-dragging::before { box-shadow:0 0 0 2px rgba(255,255,255,.8); }
.ysa__clip { align-items:center; background:#287c65; border:1px solid #60aa93; border-radius:3px; bottom:7px; color:#fff; cursor:grab; display:flex; font-size:10px; font-weight:700; min-width:8px; overflow:visible; padding:0 7px; position:absolute; text-overflow:ellipsis; top:7px; touch-action:none; white-space:nowrap; z-index:3; }
.ysa__clip[data-family="negative"] { background:#a84a4a; border-color:#d28585; }
.ysa__clip[data-family="major_penalty"] { background:#8b5d29; border-color:#c49459; }
.ysa__clip[data-family="serve_receive"] { background:#2d6392; border-color:#79acd2; }
.ysa__clip[data-family="scene"] { background:#276b76; border-color:#78b5bf; }
.ysa__clip[data-family="excluded"] { background:#60458b; border-color:#a991cf; }
.ysa__clip[data-family="excluded"]::before { background:repeating-linear-gradient(135deg,rgba(255,255,255,.16) 0,rgba(255,255,255,.16) 4px,transparent 4px,transparent 8px); content:""; inset:0; pointer-events:none; position:absolute; }
.ysa__track-row--excluded .ysa__track-label { color:#e7def5; }
.ysa__track-row--excluded .ysa__track-lane { background-color:#302c35; }
.ysa__track-row--excluded .ysa__track-lane:hover { background-color:#37313e; }
.ysa__track-row--scene .ysa__track-label { color:#d9f0f3; }
.ysa__track-row--scene .ysa__track-lane { background-color:#293437; }
.ysa__track-row--scene .ysa__track-lane:hover { background-color:#2d3b3e; }
.ysa__clip.is-selected { box-shadow:0 0 0 2px #f2c56e; z-index:5; }
.ysa__clip-anchor { background:transparent; bottom:-3px; cursor:col-resize; position:absolute; top:-3px; transform:translateX(-50%); width:15px; z-index:6; }
.ysa__clip-anchor::before { background:#ffe4a6; bottom:0; box-shadow:0 0 0 1px rgba(50,40,20,.32); content:""; left:6px; position:absolute; top:0; transition:background .12s ease,box-shadow .12s ease; width:3px; }
.ysa__clip-anchor::after { background:#ffe4a6; box-shadow:0 0 0 1px rgba(50,40,20,.38); content:""; height:8px; left:3px; position:absolute; top:-3px; transform:rotate(45deg); width:8px; }
.ysa__clip-anchor:hover::before,.ysa__clip-anchor.is-dragging::before { background:#fff1bf; box-shadow:0 0 0 2px #fff,0 0 0 3px #765a25; }
.ysa__clip-anchor:hover::after,.ysa__clip-anchor.is-dragging::after { background:#fff1bf; box-shadow:0 0 0 2px #fff,0 0 0 3px #765a25; }
.ysa__clip-handle { bottom:0; cursor:ew-resize; position:absolute; top:0; width:7px; z-index:5; }
.ysa__clip-handle[data-handle="start"] { left:-3px; }
.ysa__clip-handle[data-handle="end"] { right:-3px; }
.ysa__draft-clip { background:rgba(255,255,255,.15); border:1px dashed #d6dbd7; bottom:7px; pointer-events:none; position:absolute; top:7px; }
.ysa__events { border-top:1px solid var(--line); padding:16px 4px 4px; }
.ysa__events-heading { margin-bottom:10px; }
.ysa__events-heading h3 span { background:#e8ece8; border-radius:10px; color:#566057; font-size:11px; margin-left:6px; padding:2px 7px; }
.ysa__events-heading div { align-items:baseline; display:flex; gap:8px; }
.ysa__events-heading div span { color:var(--muted); font-size:12px; }
.ysa__events-heading strong { font-size:21px; }
.ysa__table-wrap { overflow:auto; }
.ysa table { border-collapse:collapse; font-size:12px; min-width:780px; width:100%; }
.ysa th { background:var(--soft); color:#5c625d; font-weight:700; padding:9px 10px; text-align:left; }
.ysa td { border-top:1px solid #e4e7e3; padding:9px 10px; }
.ysa tbody tr[data-event-id] { cursor:pointer; }
.ysa tbody tr[data-event-id]:hover,.ysa tbody tr.is-selected { background:#edf6f2; }
.ysa__tag { border:1px solid #aed3c5; border-radius:4px; color:#176b55; display:inline-block; font-weight:700; padding:2px 6px; }
.ysa__tag.is-negative { border-color:#e1b5b5; color:var(--danger); }
.ysa__no-events { color:var(--muted); padding:28px!important; text-align:center; }
.ysa__drawer-backdrop { background:rgba(20,23,21,.35); inset:0; opacity:0; pointer-events:none; position:fixed; transition:opacity .18s ease; z-index:40; }
.ysa__drawer-backdrop.is-open { opacity:1; pointer-events:auto; }
.ysa__drawer { background:#fff; bottom:0; box-shadow:-12px 0 32px rgba(15,20,17,.18); display:flex; flex-direction:column; max-width:92vw; position:fixed; right:0; top:0; transform:translateX(105%); transition:transform .2s ease; width:430px; z-index:41; }
.ysa__drawer.is-open { transform:translateX(0); }
.ysa__drawer-header { align-items:center; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; padding:17px 18px; }
.ysa__drawer-header div span { color:var(--muted); display:block; font-size:11px; margin-top:3px; }
.ysa__drawer-header button { align-items:center; background:transparent; border:0; border-radius:4px; color:#515651; cursor:pointer; display:flex; font-size:24px; height:32px; justify-content:center; padding:0; width:32px; }
.ysa__drawer-header button:hover { background:var(--soft); }
.ysa__storage-location { background:var(--soft); border-bottom:1px solid var(--line); display:grid; gap:3px; padding:10px 18px; }
.ysa__storage-location span { color:var(--muted); font-size:10px; font-weight:700; }
.ysa__storage-location code { color:#3f4640; font-size:11px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.ysa__session-list { flex:1; overflow:auto; }
.ysa__session-empty { color:var(--muted); font-size:12px; padding:44px 18px; text-align:center; }
.ysa__managed-session { border-bottom:1px solid var(--line); }
.ysa__managed-session summary { cursor:pointer; display:grid; gap:5px; list-style:none; padding:14px 18px; position:relative; }
.ysa__managed-session summary::-webkit-details-marker { display:none; }
.ysa__managed-session summary::after { color:#757b76; content:"›"; font-size:19px; position:absolute; right:18px; top:22px; transform:rotate(90deg); transition:transform .12s ease; }
.ysa__managed-session[open] summary::after { transform:rotate(-90deg); }
.ysa__managed-session summary strong { font-size:13px; max-width:340px; overflow:hidden; padding-right:22px; text-overflow:ellipsis; white-space:nowrap; }
.ysa__managed-session summary span { color:var(--muted); font-size:11px; }
.ysa__managed-body { background:#fafbfa; border-top:1px solid #eceeeb; padding:13px 18px 16px; }
.ysa__managed-stats { display:grid; grid-template-columns:repeat(3,1fr); margin-bottom:12px; }
.ysa__managed-stats div { border-right:1px solid var(--line); display:grid; gap:2px; padding:2px 9px; }
.ysa__managed-stats div:first-child { padding-left:0; }
.ysa__managed-stats div:last-child { border-right:0; }
.ysa__managed-stats span { color:var(--muted); font-size:10px; }
.ysa__managed-stats strong { font-size:14px; }
.ysa__managed-fields { display:grid; gap:9px; grid-template-columns:100px 1fr; }
.ysa__managed-fields label { margin:0; }
.ysa__managed-events { border:1px solid var(--line); margin-top:12px; max-height:180px; overflow:auto; }
.ysa__managed-event { align-items:center; border-bottom:1px solid #e5e8e4; display:grid; font-size:10px; gap:6px; grid-template-columns:minmax(90px,1fr) 36px minmax(112px,1fr); padding:7px 8px; }
.ysa__managed-event:last-child { border-bottom:0; }
.ysa__managed-event span:first-child { font-weight:650; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.ysa__managed-event span:nth-child(2) { text-align:right; }
.ysa__managed-event time { color:var(--muted); text-align:right; }
.ysa__managed-events-empty { color:var(--muted); font-size:10px; padding:12px; text-align:center; }
.ysa__managed-actions { display:flex; flex-wrap:wrap; gap:7px; margin-top:13px; }
.ysa__managed-actions button { min-height:31px; }
.ysa__managed-actions .ysa__managed-delete { color:var(--danger); margin-left:auto; }
.ysa__toast { background:#202321; border-radius:5px; bottom:18px; color:#fff; font-size:12px; opacity:0; padding:9px 12px; pointer-events:none; position:fixed; right:18px; transform:translateY(8px); transition:.18s ease; z-index:20; }
.ysa__toast.is-visible { opacity:1; transform:translateY(0); }
@media (max-width:900px) {
  .ysa__header { align-items:flex-start; flex-direction:column; }
  .ysa__session { grid-template-columns:1fr 1fr 1fr; }
  .ysa__identity { grid-column:1/-1; }
  .ysa__workspace { grid-template-columns:1fr; }
  .ysa__player-panel { border-right:0; grid-template-rows:minmax(300px,55vh) auto; }
  .ysa__editor { border-top:1px solid var(--line); }
}
@media (max-width:600px) {
  .ysa__header-actions { display:grid; grid-template-columns:1fr 1fr; width:100%; }
  .ysa__session { grid-template-columns:1fr 1fr; }
  .ysa__session label:nth-child(2) { grid-column:span 1; }
  .ysa__identity { grid-column:1/-1; }
  .ysa__transport { flex-wrap:wrap; }
  .ysa__transport output { order:-1; width:100%; }
  .ysa__zoom { margin-left:auto; }
  .ysa__time-grid { grid-template-columns:1fr; }
  .ysa__timeline-heading { align-items:stretch; flex-direction:column; }
  .ysa__timeline-tools { justify-content:flex-start; width:100%; }
  .ysa__timeline-heading h3 { width:100%; }
}
"""


SCORE_ANNOTATION_JS = r"""
if (element.dataset.initialized === "true") return;
element.dataset.initialized = "true";
const $ = (selector) => element.querySelector(selector);
const video = $("#ysa-video");
const videoFile = $("#ysa-video-file");
const metadataFile = $("#ysa-metadata-file");
const division = $("#ysa-division");
const judge = $("#ysa-judge");
const fpsInput = $("#ysa-fps");
const fields = $("#ysa-event-fields");
const familyButtons = [...element.querySelectorAll("#ysa-family button")];
const penalties = {restart:{name:"重启",delta:-1},discard:{name:"弃用",delta:-3},disassembly:{name:"解体",delta:-5}};
const exclusionReasons = {defocus:"画面虚焦",occlusion:"主体遮挡",corrupted_frames:"画面损坏",other:"其他"};
const sceneTypes = {irrelevant_scene:"无关场景",player_entry_exit:"选手入/离场"};
let objectUrl = null;
let currentDocument = null;
let currentStorageKey = null;
let currentStoragePath = null;
let selectedEventId = null;
let currentFamily = "positive";
let timelineScale = 1;
let pendingEventStart = null;
let pendingEventAnchor = null;
let pendingExclusionStart = null;
let selectedExclusionId = null;
let pendingSceneStart = null;
let selectedSceneId = null;
let selectedServeReceiveId = null;
let editorDirty = false;
let toastTimer = null;
let saveQueue = Promise.resolve();

const nowIso = () => new Date().toISOString();
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const numeric = (node) => Number.parseFloat(node.value || "0");
const fps = () => clamp(numeric(fpsInput) || 30, 1, 240);
const roundTime = (value) => Math.round(value * 1000) / 1000;
const snapToFrame = (value) => roundTime(Math.round(value * fps()) / fps());
const frameAt = (value) => Math.max(0, Math.round(value * fps()));
const mediaDuration = () => {
  if (Number.isFinite(video.duration) && video.duration > 0) return video.duration;
  const storedDuration = Number(currentDocument?.video?.duration_s);
  return Number.isFinite(storedDuration) && storedDuration > 0 ? storedDuration : 0;
};
const uid = () => crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const formatTime = (seconds) => {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  const millis = Math.floor((safe % 1) * 1000);
  return `${String(minutes).padStart(2,"0")}:${String(secs).padStart(2,"0")}.${String(millis).padStart(3,"0")}`;
};
const toast = (message) => {
  const node = $("#ysa-toast"); node.textContent = message; node.classList.add("is-visible");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => node.classList.remove("is-visible"), 1800);
};
const videoIdentity = (file) => `${file.name}:${file.size}:${file.lastModified}`;
const gradioFileUrl = (path) => {
  const config = window.gradio_config || {};
  const root = String(config.root || window.location.origin).replace(/\/$/, "");
  const apiPrefix = `/${String(config.api_prefix || "gradio_api").replace(/^\/+|\/+$/g, "")}`;
  return `${root}${apiPrefix}/file=${encodeURIComponent(path)}`;
};
const eventName = (event) => event.label.display_name;
const intervalsOverlap = (startA, endA, startB, endB) => startA === endA ? startA >= startB && startA <= endB : startA < endB && endA > startB;
const overlappingExclusion = (start, end, ignoredId = null) => (currentDocument?.excluded_intervals || []).find(interval => interval.exclusion_id !== ignoredId && intervalsOverlap(start, end, interval.start_s, interval.end_s));

function normalizeDocument(document) {
  document.training_data_policy = {
    ...(document.training_data_policy || {}),
    excluded_intervals_field:"excluded_intervals",
    rule:"frames_overlapping_excluded_intervals_are_ineligible"
  };
  document.serve_receive_events = Array.isArray(document.serve_receive_events) ? document.serve_receive_events : [];
  return document;
}

function blankDocument(file, sourcePath) {
  const timestamp = nowIso();
  return {
    schema_version:"yoyo_score_annotation_v2",
    annotation_id:uid(),
    revision:0,
    video:{
      video_id:uid(), file_name:file.name, file_size_bytes:file.size,
      last_modified_ms:file.lastModified, browser_identity:videoIdentity(file),
      source_path:sourcePath,
      duration_s:null, width:null, height:null,
      fingerprint:{algorithm:"sha256:first-last-1m+size", value:null}
    },
    competition:{division:division.value || "1A"},
    annotator:{judge:(judge.value || "judge1").trim() || "judge1"},
    timing_basis:{unit:"seconds", fps_assumption:fps(), frame_index_rounding:"nearest"},
    training_data_policy:{excluded_intervals_field:"excluded_intervals", rule:"frames_overlapping_excluded_intervals_are_ineligible"},
    events:[], scene_intervals:[], excluded_intervals:[], serve_receive_events:[], created_at:timestamp, updated_at:timestamp
  };
}

async function fingerprint(file) {
  const chunk = 1024 * 1024;
  const head = await file.slice(0, chunk).arrayBuffer();
  const tail = await file.slice(Math.max(0, file.size - chunk)).arrayBuffer();
  const size = new TextEncoder().encode(String(file.size));
  const merged = new Uint8Array(head.byteLength + tail.byteLength + size.byteLength);
  merged.set(new Uint8Array(head), 0); merged.set(new Uint8Array(tail), head.byteLength); merged.set(size, head.byteLength + tail.byteLength);
  const digest = await crypto.subtle.digest("SHA-256", merged);
  return [...new Uint8Array(digest)].map(value => value.toString(16).padStart(2,"0")).join("");
}

function persist(message) {
  if (!currentDocument) return Promise.resolve(null);
  currentDocument.revision = Number(currentDocument.revision || 0) + 1;
  currentDocument.updated_at = nowIso();
  currentDocument.competition.division = division.value;
  currentDocument.annotator.judge = (judge.value || "judge1").trim() || "judge1";
  currentDocument.timing_basis.fps_assumption = fps();
  const snapshot = JSON.stringify(currentDocument);
  const annotationId = currentDocument.annotation_id;
  const revision = currentDocument.revision;
  $("#ysa-session-state").textContent = `正在保存到本地文件 · 修订 ${revision}`;
  saveQueue = saveQueue
    .then(() => server.save_score_annotation(snapshot))
    .then(result => {
      if (currentDocument?.annotation_id === annotationId && currentDocument.revision >= revision) {
        currentStorageKey = result.storage_key;
        currentStoragePath = result.path;
        $("#ysa-session-state").textContent = `已保存到 ${result.path} · 修订 ${revision}`;
      }
      refreshSessionManager();
      if (message) toast(message);
      return result;
    })
    .catch(error => {
      $("#ysa-session-state").textContent = `保存失败 · 修订 ${revision}`;
      toast(`本地文件保存失败：${error?.message || error}`);
      return null;
    });
  return saveQueue;
}

async function flushCurrentSessionBeforeSwitch() {
  if (!currentDocument) return true;
  if (pendingEventStart !== null || pendingExclusionStart !== null || pendingSceneStart !== null) {
    $("#ysa-validation").textContent = "请先结束当前区间标记，再切换视频。";
    toast("当前区间尚未完成，未切换视频");
    return false;
  }
  const hadDirtyEditor = Boolean(selectedEventId && editorDirty);
  if (hadDirtyEditor && !syncSelectedFromEditor("切换前已自动保存", false)) return false;
  const result = hadDirtyEditor ? await saveQueue : await persist("切换前已自动保存");
  if (result) return true;
  toast("当前会话保存失败，未切换视频");
  return false;
}

function downloadAnnotationDocument(annotation) {
  const stem = annotation.video.file_name.replace(/\.[^.]+$/, "").replace(/[^\w\-\u4e00-\u9fff]+/g, "_");
  const blob = new Blob([JSON.stringify(annotation, null, 2) + "\n"], {type:"application/json"});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = `${stem}.score-annotation.json`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function openSessionManager() {
  $("#ysa-manager").classList.add("is-open");
  $("#ysa-manager-backdrop").classList.add("is-open");
  $("#ysa-manager").setAttribute("aria-hidden", "false");
  refreshSessionManager();
}

function closeSessionManager() {
  $("#ysa-manager").classList.remove("is-open");
  $("#ysa-manager-backdrop").classList.remove("is-open");
  $("#ysa-manager").setAttribute("aria-hidden", "true");
}

function closeActiveSession() {
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = null;
  video.pause(); video.removeAttribute("src"); video.load(); video.classList.remove("is-ready");
  $("#ysa-empty").hidden = false;
  currentDocument = null; currentStorageKey = null; currentStoragePath = null;
  selectedEventId = null; pendingEventStart = null; pendingEventAnchor = null; pendingExclusionStart = null; selectedExclusionId = null; pendingSceneStart = null; selectedSceneId = null; selectedServeReceiveId = null; editorDirty = false;
  videoFile.value = "";
  fields.disabled = true;
  ["#ysa-prev","#ysa-next","#ysa-zoom","#ysa-timeline-zoom","#ysa-new","#ysa-export","#ysa-exclusion-toggle","#ysa-exclusion-reason","#ysa-scene-toggle","#ysa-scene-type","#ysa-serve-begin","#ysa-serve-end"].forEach(selector => $(selector).disabled = true);
  $("#ysa-video-identity").textContent = "尚未选择视频";
  $("#ysa-session-state").textContent = "选择本地视频开始新的标注会话";
  clearEditor(false); renderEvents(); updateTimecode();
}

async function saveManagedSession(session, divisionValue, judgeValue) {
  const annotation = session.document;
  annotation.competition.division = divisionValue;
  annotation.annotator.judge = judgeValue.trim() || "judge1";
  annotation.revision = Number(annotation.revision || 0) + 1;
  annotation.updated_at = nowIso();
  const snapshot = JSON.stringify(annotation);
  try {
    saveQueue = saveQueue.then(() => server.save_score_annotation(snapshot));
    const result = await saveQueue;
    if (currentStorageKey === session.storage_key) {
      currentDocument = result.document;
      currentStorageKey = result.storage_key;
      currentStoragePath = result.path;
      division.value = currentDocument.competition.division;
      judge.value = currentDocument.annotator.judge;
      $("#ysa-session-state").textContent = `已保存到 ${result.path} · 修订 ${currentDocument.revision}`;
      renderEvents();
    }
    toast("会话信息已更新");
    refreshSessionManager(result.storage_key);
  } catch (error) {
    saveQueue = Promise.resolve();
    toast(`会话更新失败：${error?.message || error}`);
  }
}

async function deleteManagedSession(session) {
  const name = session.document.video?.file_name || session.storage_key;
  if (!window.confirm(`删除 ${name} 的计分标注？此操作会删除本地 JSON 文件。`)) return;
  try {
    if (currentStorageKey === session.storage_key) await saveQueue;
    await server.delete_score_annotation(session.storage_key);
    if (currentStorageKey === session.storage_key) closeActiveSession();
    toast("本地标注文件已删除");
    refreshSessionManager();
  } catch (error) {
    toast(`删除失败：${error?.message || error}`);
  }
}

async function refreshSessionManager(expandedKey = null) {
  let sessions = [];
  try { sessions = await server.list_score_annotations(); }
  catch (error) { toast(`读取会话列表失败：${error?.message || error}`); }
  $("#ysa-manager-count").textContent = sessions.length;
  const list = $("#ysa-session-list");
  if (!sessions.length) {
    list.innerHTML = '<div class="ysa__session-empty">尚无本地计分标注</div>';
    return;
  }
  list.innerHTML = sessions.map(session => {
    const annotation = session.document;
    const events = annotation.events || [];
    const total = events.reduce((sum,event) => sum + Number(event.label?.score_delta || 0), 0);
    const updated = annotation.updated_at ? new Date(annotation.updated_at).toLocaleString() : "-";
    const eventRows = events.length ? events.map(event => `<div class="ysa__managed-event"><span>${escapeHtml(event.label?.display_name || event.label?.family || "事件")}</span><span>${event.label?.score_delta > 0 ? "+" : ""}${Number(event.label?.score_delta || 0)}</span><time>${formatTime(event.timing?.evidence_start_s)}–${formatTime(event.timing?.evidence_end_s)}</time></div>`).join("") : '<div class="ysa__managed-events-empty">尚无计分事件</div>';
    return `<details class="ysa__managed-session" data-storage-key="${escapeHtml(session.storage_key)}"${session.storage_key === expandedKey ? " open" : ""}><summary><strong>${escapeHtml(annotation.video?.file_name || session.storage_key)}</strong><span>${escapeHtml(annotation.competition?.division || "1A")} · ${escapeHtml(annotation.annotator?.judge || "judge1")} · ${events.length} 个事件 · ${escapeHtml(updated)}</span></summary><div class="ysa__managed-body"><div class="ysa__managed-stats"><div><span>事件</span><strong>${events.length}</strong></div><div><span>总分</span><strong>${total}</strong></div><div><span>修订</span><strong>${Number(annotation.revision || 0)}</strong></div></div><div class="ysa__managed-fields"><label>组别<select data-managed-division>${["1A","2A","3A","4A","5A"].map(value => `<option${value === annotation.competition?.division ? " selected" : ""}>${value}</option>`).join("")}</select></label><label>裁判<input data-managed-judge maxlength="80" value="${escapeHtml(annotation.annotator?.judge || "judge1")}"></label></div><div class="ysa__managed-events">${eventRows}</div><div class="ysa__managed-actions"><button type="button" class="ysa__button ysa__button--primary" data-managed-continue>继续标注</button><button type="button" class="ysa__button" data-managed-export>导出</button><button type="button" class="ysa__button ysa__managed-delete" data-managed-delete>删除</button></div></div></details>`;
  }).join("");
  element.querySelectorAll(".ysa__managed-session").forEach((details, index) => {
    const session = sessions[index];
    const divisionInput = details.querySelector("[data-managed-division]");
    const judgeInput = details.querySelector("[data-managed-judge]");
    divisionInput.addEventListener("change", () => saveManagedSession(session, divisionInput.value, judgeInput.value));
    judgeInput.addEventListener("change", () => saveManagedSession(session, divisionInput.value, judgeInput.value));
    details.querySelector("[data-managed-continue]").addEventListener("click", () => resumeManagedSession(session));
    details.querySelector("[data-managed-export]").addEventListener("click", () => { downloadAnnotationDocument(session.document); toast("元数据已导出"); });
    details.querySelector("[data-managed-delete]").addEventListener("click", () => deleteManagedSession(session));
  });
}

function setFamily(family) {
  currentFamily = family;
  familyButtons.forEach(button => button.classList.toggle("is-active", button.dataset.family === family));
  const score = $("#ysa-score");
  $("#ysa-score-row").hidden = family === "major_penalty";
  $("#ysa-penalty-row").hidden = family !== "major_penalty";
  if (family === "positive") { score.min = "0"; score.max = "10"; score.value = String(clamp(Math.abs(numeric(score) || 1), 0, 10)); }
  if (family === "negative") { score.min = "-10"; score.max = "-1"; score.value = String(-clamp(Math.abs(numeric(score) || 1), 1, 10)); }
}

function clearEditor(render = true) {
  selectedEventId = null;
  selectedExclusionId = null;
  selectedSceneId = null;
  pendingEventStart = null;
  pendingEventAnchor = null;
  editorDirty = false;
  $("#ysa-editor-title").textContent = "新建计分事件";
  $("#ysa-save").textContent = "记录起点";
  $("#ysa-save").hidden = false;
  $("#ysa-delete").hidden = true;
  $("#ysa-action").value = "";
  const t = roundTime(video.currentTime || 0);
  $("#ysa-anchor").value = t; $("#ysa-start").value = t; $("#ysa-end").value = t;
  $("#ysa-validation").textContent = "";
  updateExclusionControls();
  updateSceneControls();
  if (render) renderEvents();
}

function populateEditor(event) {
  selectedExclusionId = null;
  selectedSceneId = null;
  pendingEventStart = null;
  pendingEventAnchor = null;
  editorDirty = false;
  setFamily(event.label.family);
  $("#ysa-score").value = event.label.score_delta;
  if (event.label.penalty_type) $("#ysa-penalty").value = event.label.penalty_type;
  $("#ysa-anchor").value = event.timing.anchor_s;
  $("#ysa-start").value = event.timing.evidence_start_s;
  $("#ysa-end").value = event.timing.evidence_end_s;
  $("#ysa-action").value = event.action_name || "";
  $("#ysa-editor-title").textContent = "编辑计分事件";
  $("#ysa-save").hidden = true;
  $("#ysa-delete").hidden = false;
  updateExclusionControls();
  updateSceneControls();
}

function loadEvent(id, seekToAnchor = true) {
  if (selectedEventId && selectedEventId !== id && !syncSelectedFromEditor("事件已自动更新", false)) return;
  const event = currentDocument?.events.find(item => item.event_id === id);
  if (!event) return;
  selectedEventId = id;
  populateEditor(event);
  if (seekToAnchor) video.currentTime = event.timing.anchor_s;
  renderEvents();
}

function updateEventTiming(event, start, anchor, end) {
  start = snapToFrame(start); anchor = snapToFrame(anchor); end = snapToFrame(end);
  event.timing.evidence_start_s = roundTime(start);
  event.timing.anchor_s = roundTime(anchor);
  event.timing.evidence_end_s = roundTime(end);
  event.timing.evidence_start_frame_index = frameAt(start);
  event.timing.anchor_frame_index = frameAt(anchor);
  event.timing.evidence_end_frame_index = frameAt(end);
  event.timing.fps_assumption = fps();
  event.updated_at = nowIso();
  if (event.event_id === selectedEventId) {
    $("#ysa-start").value = event.timing.evidence_start_s;
    $("#ysa-anchor").value = event.timing.anchor_s;
    $("#ysa-end").value = event.timing.evidence_end_s;
  }
}

function buildExclusion(start, end, existing = null) {
  start = snapToFrame(start); end = snapToFrame(end);
  if (start > end) [start, end] = [end, start];
  if (end - start < 1 / fps() - .0005) {
    toast("不可标记区间至少需要一帧");
    return null;
  }
  const timestamp = nowIso();
  return {
    exclusion_id:existing?.exclusion_id || uid(),
    start_s:roundTime(start), end_s:roundTime(end),
    start_frame_index:frameAt(start), end_frame_index:frameAt(end),
    fps_assumption:fps(), reason:existing?.reason || $("#ysa-exclusion-reason").value,
    reason_detail:existing?.reason_detail || null, training_eligible:false,
    created_at:existing?.created_at || timestamp, updated_at:timestamp
  };
}

function updateExclusionControls() {
  const button = $("#ysa-exclusion-toggle");
  button.disabled = !currentDocument || pendingSceneStart !== null;
  button.classList.toggle("is-recording", pendingExclusionStart !== null);
  const selectedDeletableId = selectedExclusionId !== null || selectedSceneId !== null || selectedServeReceiveId !== null;
  button.classList.toggle("is-selected", selectedDeletableId && pendingExclusionStart === null);
  button.textContent = pendingExclusionStart !== null ? "结束并标记不可用" : selectedDeletableId ? "删除所选不可用区间" : "标记不可用起点";
  $("#ysa-exclusion-reason").disabled = !currentDocument || pendingExclusionStart !== null || selectedExclusionId !== null || pendingSceneStart !== null;
}

function selectExclusion(id, seek = true) {
  if (selectedEventId && !syncSelectedFromEditor("事件已自动更新", false)) return;
  clearEditor(false);
  selectedExclusionId = id;
  const interval = currentDocument?.excluded_intervals.find(item => item.exclusion_id === id);
  if (!interval) { selectedExclusionId = null; updateExclusionControls(); return; }
  $("#ysa-exclusion-reason").value = interval.reason;
  if (seek) video.currentTime = interval.start_s;
  updateExclusionControls(); renderEvents();
}

function addExclusion(start, end, message = "不可标记区间已添加") {
  const interval = buildExclusion(start, end);
  if (!interval) return null;
  currentDocument.excluded_intervals.push(interval);
  pendingExclusionStart = null;
  selectedExclusionId = interval.exclusion_id;
  updateExclusionControls(); updateSceneControls();
  persist(message); renderEvents();
  return interval;
}

function buildSceneInterval(start, end, existing = null) {
  start = snapToFrame(start); end = snapToFrame(end);
  if (start > end) [start, end] = [end, start];
  if (end - start < 1 / fps() - .0005) {
    toast("场景区间至少需要一帧");
    return null;
  }
  const timestamp = nowIso();
  return {
    scene_interval_id:existing?.scene_interval_id || uid(),
    start_s:roundTime(start), end_s:roundTime(end),
    start_frame_index:frameAt(start), end_frame_index:frameAt(end),
    fps_assumption:fps(), scene_type:existing?.scene_type || $("#ysa-scene-type").value,
    created_at:existing?.created_at || timestamp, updated_at:timestamp
  };
}

function updateSceneControls() {
  const button = $("#ysa-scene-toggle");
  button.disabled = !currentDocument || pendingExclusionStart !== null;
  button.classList.toggle("is-recording", pendingSceneStart !== null);
  button.classList.toggle("is-selected", selectedSceneId !== null && pendingSceneStart === null);
  button.textContent = pendingSceneStart !== null ? "结束并标记场景" : selectedSceneId !== null ? "删除所选场景区间" : "标记场景起点";
  $("#ysa-scene-type").disabled = !currentDocument || pendingSceneStart !== null || selectedSceneId !== null || pendingExclusionStart !== null;
}

function selectSceneInterval(id, seek = true) {
  if (selectedEventId && !syncSelectedFromEditor("事件已自动更新", false)) return;
  clearEditor(false);
  selectedSceneId = id;
  const interval = currentDocument?.scene_intervals.find(item => item.scene_interval_id === id);
  if (!interval) { selectedSceneId = null; updateSceneControls(); return; }
  $("#ysa-scene-type").value = interval.scene_type;
  if (seek) video.currentTime = interval.start_s;
  updateSceneControls(); renderEvents();
}

function addSceneInterval(start, end, message = "场景区间已添加") {
  const interval = buildSceneInterval(start, end);
  if (!interval) return null;
  currentDocument.scene_intervals.push(interval);
  pendingSceneStart = null;
  selectedSceneId = interval.scene_interval_id;
  updateSceneControls(); updateExclusionControls();
  persist(message); renderEvents();
  return interval;
}

function buildEvent() {
  const start = roundTime(numeric($("#ysa-start")));
  const anchor = roundTime(numeric($("#ysa-anchor")));
  const end = roundTime(numeric($("#ysa-end")));
  if (start < 0 || anchor < 0 || end < 0 || start > anchor || anchor > end) {
    $("#ysa-validation").textContent = "时间必须满足 Evidence 起点 ≤ Anchor ≤ Evidence 终点。";
    return null;
  }
  if (mediaDuration() && end > mediaDuration() + .001) {
    $("#ysa-validation").textContent = "Evidence 终点不能超过视频时长。";
    return null;
  }
  const exclusion = overlappingExclusion(start, end);
  if (exclusion) {
    $("#ysa-validation").textContent = `该 Evidence 区间与不可标记片段重叠（${exclusionReasons[exclusion.reason]}）。`;
    return null;
  }
  let scoreDelta = Number.parseInt($("#ysa-score").value, 10);
  let displayName = currentFamily === "positive" ? "正向计分" : "负向计分";
  let penaltyType = null;
  if (currentFamily === "positive") scoreDelta = clamp(Number.isFinite(scoreDelta) ? scoreDelta : 0, 0, 10);
  if (currentFamily === "negative") scoreDelta = clamp(Number.isFinite(scoreDelta) ? scoreDelta : -1, -10, -1);
  if (currentFamily === "major_penalty") {
    penaltyType = $("#ysa-penalty").value;
    scoreDelta = penalties[penaltyType].delta;
    displayName = `重点扣分 · ${penalties[penaltyType].name}`;
  }
  const existing = currentDocument.events.find(item => item.event_id === selectedEventId);
  const anchorSource = existing ? existing.timing.anchor_source : (pendingEventAnchor === null ? "evidence_end_default" : "manual");
  const timestamp = nowIso();
  return {
    event_id:existing?.event_id || uid(), sequence_index:0,
    label:{family:currentFamily, code:currentFamily === "major_penalty" ? `major_penalty_${penaltyType}` : `score_${currentFamily}`, display_name:displayName, score_delta:scoreDelta, penalty_type:penaltyType},
    timing:{anchor_s:anchor, anchor_source:anchorSource, evidence_start_s:start, evidence_end_s:end, anchor_frame_index:frameAt(anchor), evidence_start_frame_index:frameAt(start), evidence_end_frame_index:frameAt(end), fps_assumption:fps(), boundary_uncertainty_allowed:true},
    action_name:$("#ysa-action").value.trim() || null,
    created_at:existing?.created_at || timestamp, updated_at:timestamp
  };
}

function syncSelectedFromEditor(message = "事件已自动更新", render = true) {
  if (!selectedEventId || !currentDocument || !editorDirty) return true;
  const event = buildEvent();
  if (!event) return false;
  const index = currentDocument.events.findIndex(item => item.event_id === selectedEventId);
  if (index < 0) return false;
  currentDocument.events[index] = event;
  editorDirty = false;
  persist(message);
  if (render) renderEvents();
  return true;
}

function setAnchorFromCurrent() {
  if (!currentDocument) return;
  const current = snapToFrame(video.currentTime || 0);
  const start = roundTime(numeric($("#ysa-start")));
  const end = roundTime(numeric($("#ysa-end")));
  if (pendingEventStart !== null) {
    $("#ysa-anchor").value = current;
    $("#ysa-validation").textContent = "";
    pendingEventAnchor = current;
    return;
  }
  if (selectedEventId && (current < start || current > end)) {
    $("#ysa-validation").textContent = "Anchor 必须位于当前 Evidence 区间内。";
    return;
  }
  $("#ysa-anchor").value = current;
  $("#ysa-validation").textContent = "";
  if (!selectedEventId) return;
  const selectedEvent = currentDocument.events.find(event => event.event_id === selectedEventId);
  if (selectedEvent) selectedEvent.timing.anchor_source = "manual";
  editorDirty = true;
  if (syncSelectedFromEditor("Anchor 已更新", false)) renderEvents();
}

function renderEvents() {
  const events = currentDocument?.events || [];
  events.sort((a,b) => a.timing.anchor_s - b.timing.anchor_s || a.created_at.localeCompare(b.created_at));
  events.forEach((event,index) => event.sequence_index = index + 1);
  $("#ysa-event-count").textContent = events.length;
  $("#ysa-total-score").textContent = events.reduce((sum,event) => sum + event.label.score_delta, 0);
  $("#ysa-event-list").innerHTML = events.length ? events.map(event => {
    const negative = event.label.score_delta < 0 ? " is-negative" : "";
    return `<tr data-event-id="${escapeHtml(event.event_id)}" class="${event.event_id === selectedEventId ? "is-selected" : ""}"><td>${event.sequence_index}</td><td><span class="ysa__tag${negative}">${escapeHtml(eventName(event))}</span></td><td>${event.label.score_delta > 0 ? "+" : ""}${event.label.score_delta}</td><td>${formatTime(event.timing.anchor_s)} · f${event.timing.anchor_frame_index}</td><td>${formatTime(event.timing.evidence_start_s)} – ${formatTime(event.timing.evidence_end_s)}</td><td>${escapeHtml(event.action_name || "—")}</td><td>${new Date(event.updated_at).toLocaleTimeString()}</td></tr>`;
  }).join("") : '<tr><td colspan="7" class="ysa__no-events">尚无计分事件</td></tr>';
  element.querySelectorAll("#ysa-event-list tr[data-event-id]").forEach(row => row.addEventListener("click", () => loadEvent(row.dataset.eventId)));
  renderTimeline();
}

function renderTimeline() {
  const duration = mediaDuration();
  const events = currentDocument?.events || [];
  const content = $("#ysa-timeline-content");
  content.style.width = `${timelineScale * 100}%`;
  element.querySelectorAll(".ysa__track-lane").forEach(lane => {
    lane.querySelectorAll(".ysa__clip,.ysa__draft-clip").forEach(node => node.remove());
  });
  const ruler = $("#ysa-ruler-lane");
  ruler.innerHTML = "";
  if (!duration) { updateTimecode(); return; }
  const tickCount = Math.max(10, Math.ceil(timelineScale * 10));
  for (let index = 0; index <= tickCount; index += 1) {
    const tick = document.createElement("span");
    tick.className = "ysa__tick";
    tick.style.left = `${100 * index / tickCount}%`;
    tick.textContent = formatTime(duration * index / tickCount).slice(0, 8);
    ruler.append(tick);
  }
  events.forEach(event => {
    const lane = element.querySelector(`[data-track="${event.label.family}"]`);
    if (!lane) return;
    const clip = document.createElement("div");
    clip.className = `ysa__clip${event.event_id === selectedEventId ? " is-selected" : ""}`;
    clip.dataset.eventId = event.event_id;
    clip.dataset.family = event.label.family;
    clip.title = `${event.label.display_name} ${event.label.score_delta} · ${formatTime(event.timing.evidence_start_s)}–${formatTime(event.timing.evidence_end_s)}`;
    clip.innerHTML = `<span>${event.label.score_delta > 0 ? "+" : ""}${event.label.score_delta}</span><i class="ysa__clip-handle" data-handle="start"></i><i class="ysa__clip-anchor" data-handle="anchor" title="拖动调整 Anchor"></i><i class="ysa__clip-handle" data-handle="end"></i>`;
    updateClipGeometry(clip, event, duration);
    clip.addEventListener("pointerdown", pointerEvent => beginClipDrag(pointerEvent, event, clip));
    lane.append(clip);
  });
  (currentDocument?.excluded_intervals || []).forEach(interval => {
    const lane = element.querySelector('[data-track="excluded"]');
    if (!lane) return;
    const clip = document.createElement("div");
    clip.className = `ysa__clip${interval.exclusion_id === selectedExclusionId ? " is-selected" : ""}`;
    clip.dataset.exclusionId = interval.exclusion_id;
    clip.dataset.family = "excluded";
    clip.title = `${exclusionReasons[interval.reason] || "不可标记"} · ${formatTime(interval.start_s)}–${formatTime(interval.end_s)} · 不用于训练`;
    clip.innerHTML = `<span>${escapeHtml(exclusionReasons[interval.reason] || "不可标记")}</span><i class="ysa__clip-handle" data-handle="start"></i><i class="ysa__clip-handle" data-handle="end"></i>`;
    updateExclusionGeometry(clip, interval, duration);
    clip.addEventListener("pointerdown", pointerEvent => beginExclusionDrag(pointerEvent, interval, clip));
    lane.append(clip);
  });
  (currentDocument?.scene_intervals || []).forEach(interval => {
    const lane = element.querySelector('[data-track="scene"]');
    if (!lane) return;
    const clip = document.createElement("div");
    clip.className = `ysa__clip${interval.scene_interval_id === selectedSceneId ? " is-selected" : ""}`;
    clip.dataset.sceneIntervalId = interval.scene_interval_id;
    clip.dataset.family = "scene";
    clip.title = `${sceneTypes[interval.scene_type] || "场景"} · ${formatTime(interval.start_s)}–${formatTime(interval.end_s)}`;
    clip.innerHTML = `<span>${escapeHtml(sceneTypes[interval.scene_type] || "场景")}</span><i class="ysa__clip-handle" data-handle="start"></i><i class="ysa__clip-handle" data-handle="end"></i>`;
    updateExclusionGeometry(clip, interval, duration);
    clip.addEventListener("pointerdown", pointerEvent => beginSceneDrag(pointerEvent, interval, clip));
    lane.append(clip);
  });
  (currentDocument?.serve_receive_events || []).forEach(event => {
    const lane = element.querySelector('[data-track="serve_receive"]');
    if (!lane) return;
    const clip = document.createElement("div");
    clip.className = `ysa__clip ysa__clip--serve-receive${event.event_id === selectedServeReceiveId ? " is-selected" : ""}`;
    clip.dataset.serveReceiveId = event.event_id;
    clip.dataset.family = "serve_receive";
    clip.title = `${event.marker.type} · ${formatTime(event.timing.evidence_start_s)}–${formatTime(event.timing.evidence_end_s)}`;
    clip.innerHTML = `<span>${event.marker.type}</span>`;
    updateExclusionGeometry(clip, {start_s:event.timing.evidence_start_s,end_s:event.timing.evidence_end_s}, duration);
    clip.addEventListener("click", pointerEvent => {
      pointerEvent.stopPropagation();
      selectedServeReceiveId = event.event_id;
      selectedEventId = null; selectedExclusionId = null; selectedSceneId = null;
      updateExclusionControls(); renderTimeline();
    });
    lane.append(clip);
  });
  updateTimecode();
}

function updateClipGeometry(clip, event, duration) {
  const start = event.timing.evidence_start_s;
  const end = event.timing.evidence_end_s;
  clip.style.left = `${100 * start / duration}%`;
  clip.style.width = `${Math.max(0, 100 * (end - start) / duration)}%`;
  const anchorRatio = end > start ? 100 * (event.timing.anchor_s - start) / (end - start) : 50;
  clip.querySelector(".ysa__clip-anchor").style.left = `${clamp(anchorRatio, 0, 100)}%`;
}

function updateExclusionGeometry(clip, interval, duration) {
  clip.style.left = `${100 * interval.start_s / duration}%`;
  clip.style.width = `${Math.max(0, 100 * (interval.end_s - interval.start_s) / duration)}%`;
}

function beginExclusionDrag(pointerEvent, exclusion, clip) {
  pointerEvent.preventDefault(); pointerEvent.stopPropagation();
  if (selectedEventId && !syncSelectedFromEditor("事件已自动更新", false)) return;
  exclusion = currentDocument.excluded_intervals.find(interval => interval.exclusion_id === exclusion.exclusion_id) || exclusion;
  const mode = pointerEvent.target.dataset.handle || "move";
  const lane = clip.parentElement;
  const duration = mediaDuration();
  const origin = {...exclusion};
  const startX = pointerEvent.clientX;
  let moved = false;
  selectedExclusionId = exclusion.exclusion_id; selectedEventId = null; selectedSceneId = null; selectedServeReceiveId = null;
  $("#ysa-exclusion-reason").value = exclusion.reason;
  updateExclusionControls(); updateSceneControls();
  element.querySelectorAll(".ysa__clip,#ysa-event-list tr").forEach(node => node.classList.toggle("is-selected", node.dataset.exclusionId === selectedExclusionId));
  clip.setPointerCapture(pointerEvent.pointerId);
  const onMove = moveEvent => {
    const delta = (moveEvent.clientX - startX) / lane.getBoundingClientRect().width * duration;
    moved = moved || Math.abs(moveEvent.clientX - startX) >= 2;
    let start = origin.start_s;
    let end = origin.end_s;
    if (mode === "move") {
      const length = end - start;
      const shift = clamp(delta, -start, duration - end);
      start += shift; end = start + length;
    } else if (mode === "start") start = clamp(start + delta, 0, end - 1 / fps());
    else if (mode === "end") end = clamp(end + delta, start + 1 / fps(), duration);
    const updated = buildExclusion(start, end, exclusion);
    if (!updated) return;
    Object.assign(exclusion, updated);
    updateExclusionGeometry(clip, exclusion, duration);
  };
  const onUp = () => {
    clip.removeEventListener("pointermove", onMove); clip.removeEventListener("pointerup", onUp); clip.removeEventListener("pointercancel", onUp);
    if (moved) { persist("不可标记区间已更新"); renderEvents(); }
    else selectExclusion(exclusion.exclusion_id, true);
  };
  clip.addEventListener("pointermove", onMove); clip.addEventListener("pointerup", onUp); clip.addEventListener("pointercancel", onUp);
}

function beginSceneDrag(pointerEvent, sceneInterval, clip) {
  pointerEvent.preventDefault(); pointerEvent.stopPropagation();
  if (selectedEventId && !syncSelectedFromEditor("事件已自动更新", false)) return;
  sceneInterval = currentDocument.scene_intervals.find(interval => interval.scene_interval_id === sceneInterval.scene_interval_id) || sceneInterval;
  const mode = pointerEvent.target.dataset.handle || "move";
  const lane = clip.parentElement;
  const duration = mediaDuration();
  const origin = {...sceneInterval};
  const startX = pointerEvent.clientX;
  let moved = false;
  selectedSceneId = sceneInterval.scene_interval_id; selectedEventId = null; selectedExclusionId = null; selectedServeReceiveId = null;
  $("#ysa-scene-type").value = sceneInterval.scene_type;
  updateSceneControls(); updateExclusionControls();
  element.querySelectorAll(".ysa__clip,#ysa-event-list tr").forEach(node => node.classList.toggle("is-selected", node.dataset.sceneIntervalId === selectedSceneId));
  clip.setPointerCapture(pointerEvent.pointerId);
  const onMove = moveEvent => {
    const delta = (moveEvent.clientX - startX) / lane.getBoundingClientRect().width * duration;
    moved = moved || Math.abs(moveEvent.clientX - startX) >= 2;
    let start = origin.start_s;
    let end = origin.end_s;
    if (mode === "move") {
      const length = end - start;
      const shift = clamp(delta, -start, duration - end);
      start += shift; end = start + length;
    } else if (mode === "start") start = clamp(start + delta, 0, end - 1 / fps());
    else if (mode === "end") end = clamp(end + delta, start + 1 / fps(), duration);
    const updated = buildSceneInterval(start, end, sceneInterval);
    if (!updated) return;
    Object.assign(sceneInterval, updated);
    updateExclusionGeometry(clip, sceneInterval, duration);
  };
  const onUp = () => {
    clip.removeEventListener("pointermove", onMove); clip.removeEventListener("pointerup", onUp); clip.removeEventListener("pointercancel", onUp);
    if (moved) { persist("场景区间已更新"); renderEvents(); }
    else selectSceneInterval(sceneInterval.scene_interval_id, true);
  };
  clip.addEventListener("pointermove", onMove); clip.addEventListener("pointerup", onUp); clip.addEventListener("pointercancel", onUp);
}

function beginClipDrag(pointerEvent, scoreEvent, clip) {
  pointerEvent.preventDefault(); pointerEvent.stopPropagation();
  if (selectedEventId && !syncSelectedFromEditor("事件已自动更新", false)) return;
  scoreEvent = currentDocument.events.find(event => event.event_id === scoreEvent.event_id) || scoreEvent;
  const mode = pointerEvent.target.dataset.handle || "move";
  const lane = clip.parentElement;
  const duration = mediaDuration();
  const origin = {...scoreEvent.timing};
  const startX = pointerEvent.clientX;
  let moved = false;
  selectedEventId = scoreEvent.event_id;
  populateEditor(scoreEvent);
  if (mode === "anchor") pointerEvent.target.classList.add("is-dragging");
  element.querySelectorAll(".ysa__clip,#ysa-event-list tr").forEach(node => node.classList.toggle("is-selected", node.dataset.eventId === selectedEventId));
  clip.setPointerCapture(pointerEvent.pointerId);
  const onMove = moveEvent => {
    const delta = (moveEvent.clientX - startX) / lane.getBoundingClientRect().width * duration;
    moved = moved || Math.abs(moveEvent.clientX - startX) >= 2;
    let start = origin.evidence_start_s;
    let anchor = origin.anchor_s;
    let end = origin.evidence_end_s;
    if (mode === "move") {
      const length = end - start;
      const shift = clamp(delta, -start, duration - end);
      start += shift; anchor += shift; end = start + length;
    } else if (mode === "start") start = clamp(start + delta, 0, anchor);
    else if (mode === "end") end = clamp(end + delta, anchor, duration);
    else if (mode === "anchor") anchor = clamp(anchor + delta, start, end);
    updateEventTiming(scoreEvent, start, anchor, end);
    updateClipGeometry(clip, scoreEvent, duration);
  };
  const onUp = () => {
    clip.removeEventListener("pointermove", onMove); clip.removeEventListener("pointerup", onUp); clip.removeEventListener("pointercancel", onUp);
    clip.querySelector(".ysa__clip-anchor")?.classList.remove("is-dragging");
    if (moved) {
      if (overlappingExclusion(scoreEvent.timing.evidence_start_s, scoreEvent.timing.evidence_end_s)) {
        Object.assign(scoreEvent.timing, origin);
        populateEditor(scoreEvent);
        $("#ysa-validation").textContent = "计分事件不能移动到不可标记片段内。";
        toast("与不可标记区间冲突，已撤销移动");
        renderEvents();
        return;
      }
      if (mode === "anchor") scoreEvent.timing.anchor_source = "manual";
      persist("轨道事件已更新"); renderEvents();
    }
    else loadEvent(scoreEvent.event_id, true);
  };
  clip.addEventListener("pointermove", onMove); clip.addEventListener("pointerup", onUp); clip.addEventListener("pointercancel", onUp);
}

function beginPlayheadDrag(pointerEvent) {
  pointerEvent.preventDefault(); pointerEvent.stopPropagation();
  const duration = mediaDuration();
  if (!currentDocument || !duration) return;
  if (selectedEventId && !syncSelectedFromEditor("事件已自动更新", false)) return;
  const playhead = pointerEvent.currentTarget;
  const lane = playhead.parentElement;
  const rect = lane.getBoundingClientRect();
  const seek = clientX => {
    video.currentTime = clamp((clientX - rect.left) / rect.width, 0, 1) * duration;
  };
  playhead.classList.add("is-dragging");
  playhead.setPointerCapture(pointerEvent.pointerId);
  seek(pointerEvent.clientX);
  const onMove = moveEvent => seek(moveEvent.clientX);
  const onUp = () => {
    playhead.removeEventListener("pointermove", onMove); playhead.removeEventListener("pointerup", onUp); playhead.removeEventListener("pointercancel", onUp);
    playhead.classList.remove("is-dragging");
    renderEvents();
  };
  playhead.addEventListener("pointermove", onMove); playhead.addEventListener("pointerup", onUp); playhead.addEventListener("pointercancel", onUp);
}

function beginTrackDraft(pointerEvent) {
  const lane = pointerEvent.currentTarget;
  const duration = mediaDuration();
  if (pointerEvent.target !== lane || !currentDocument || !duration) return;
  pointerEvent.preventDefault();
  const rect = lane.getBoundingClientRect();
  const at = clientX => clamp((clientX - rect.left) / rect.width, 0, 1) * duration;
  const startTime = at(pointerEvent.clientX);
  const startX = pointerEvent.clientX;
  const draft = document.createElement("div");
  draft.className = "ysa__draft-clip"; lane.append(draft);
  lane.setPointerCapture(pointerEvent.pointerId);
  const onMove = moveEvent => {
    const current = at(moveEvent.clientX);
    draft.style.left = `${100 * Math.min(startTime, current) / duration}%`;
    draft.style.width = `${100 * Math.abs(current - startTime) / duration}%`;
  };
  const onUp = upEvent => {
    lane.removeEventListener("pointermove", onMove); lane.removeEventListener("pointerup", onUp); lane.removeEventListener("pointercancel", onUp); draft.remove();
    const endTime = at(upEvent.clientX);
    if (selectedEventId && !syncSelectedFromEditor("事件已自动更新", false)) return;
    clearEditor(false);
    if (Math.abs(upEvent.clientX - startX) < 4) { video.currentTime = endTime; renderEvents(); return; }
    const start = snapToFrame(Math.min(startTime, endTime));
    const end = snapToFrame(Math.max(startTime, endTime));
    if (lane.dataset.track === "excluded") {
      addExclusion(start, end, "已从轨道添加不可标记区间");
      return;
    }
    if (lane.dataset.track === "scene") {
      addSceneInterval(start, end, "已从轨道添加场景区间");
      return;
    }
    setFamily(lane.dataset.track);
    if (lane.dataset.track === "positive") $("#ysa-score").value = "1";
    if (lane.dataset.track === "negative") $("#ysa-score").value = "-1";
    if (lane.dataset.track === "major_penalty") $("#ysa-penalty").value = "restart";
    $("#ysa-start").value = start; $("#ysa-anchor").value = end; $("#ysa-end").value = end; $("#ysa-action").value = "";
    const event = buildEvent();
    if (event) { currentDocument.events.push(event); persist("已从轨道添加事件"); loadEvent(event.event_id, true); }
  };
  lane.addEventListener("pointermove", onMove); lane.addEventListener("pointerup", onUp); lane.addEventListener("pointercancel", onUp);
}

function updateTimecode() {
  const time = video.currentTime || 0;
  $("#ysa-timecode").textContent = `${formatTime(time)} · f${frameAt(time)}`;
  const duration = mediaDuration();
  const ratio = duration ? clamp(time / duration, 0, 1) : 0;
  element.querySelectorAll(".ysa__track-playhead").forEach(node => {
    node.style.left = `${ratio * 100}%`;
    node.setAttribute("aria-valuenow", String(roundTime(time)));
    node.setAttribute("aria-valuemax", String(roundTime(duration)));
  });
}

function activateScoreSession(stored, videoSource, identityLabel) {
  currentStorageKey = stored?.storage_key || null;
  currentStoragePath = stored?.path || null;
  currentDocument = normalizeDocument(stored.document);
  division.value = currentDocument.competition?.division || "1A";
  judge.value = currentDocument.annotator?.judge || "judge1";
  fpsInput.value = currentDocument.timing_basis?.fps_assumption || 30;
  video.src = videoSource;
  video.classList.add("is-ready"); $("#ysa-empty").hidden = true;
  fields.disabled = false; $("#ysa-prev").disabled = false; $("#ysa-next").disabled = false; $("#ysa-zoom").disabled = false; $("#ysa-timeline-zoom").disabled = false; $("#ysa-new").disabled = false; $("#ysa-export").disabled = false; $("#ysa-exclusion-toggle").disabled = false; $("#ysa-exclusion-reason").disabled = false; $("#ysa-scene-toggle").disabled = false; $("#ysa-scene-type").disabled = false; $("#ysa-serve-begin").disabled = false; $("#ysa-serve-end").disabled = false;
  $("#ysa-video-identity").textContent = identityLabel;
  clearEditor();
  $("#ysa-session-state").textContent = `已从 ${stored.path} 恢复 ${currentDocument.events.length} 个事件、${currentDocument.scene_intervals.length} 个场景区间、${currentDocument.excluded_intervals.length} 个不可标记区间 · 修订 ${currentDocument.revision}`;
}

async function resumeManagedSession(session) {
  if (!await flushCurrentSessionBeforeSwitch()) return;
  try {
    const stored = await server.load_score_annotation_session(session.storage_key);
    if (!stored.video_path) {
      toast(stored.video_error || "该计分会话对应的视频不可用");
      return;
    }
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
    closeSessionManager();
    videoFile.value = "";
    const videoInfo = stored.document.video || {};
    const sizeMb = Number(videoInfo.file_size_bytes || 0) / 1048576;
    activateScoreSession(
      stored,
      gradioFileUrl(stored.video_path),
      `${videoInfo.file_name || "视频"} · ${sizeMb.toFixed(1)} MB`,
    );
  } catch (error) {
    toast(`继续标注失败：${error?.message || error}`);
  }
}

async function openVideo(file) {
  if (!file) return;
  const identity = videoIdentity(file);
  if (!await flushCurrentSessionBeforeSwitch()) {
    videoFile.value = "";
    return;
  }
  let stored = null;
  try { stored = await server.load_score_annotation(identity); }
  catch (error) { toast(`读取本地标注失败：${error?.message || error}`); }
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  if (stored) {
    activateScoreSession(stored, objectUrl, `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`);
  } else {
    let sourcePath;
    try {
      sourcePath = await server.resolve_score_video_source({
        file_name:file.name,
        file_size_bytes:file.size,
        last_modified_ms:file.lastModified,
      });
    } catch (error) {
      URL.revokeObjectURL(objectUrl); objectUrl = null; videoFile.value = "";
      toast(`无法登记视频源：${error?.message || error}`);
      return;
    }
    currentStorageKey = null; currentStoragePath = null; currentDocument = normalizeDocument(blankDocument(file, sourcePath));
    division.value = currentDocument.competition.division; judge.value = currentDocument.annotator.judge;
    video.src = objectUrl;
    video.classList.add("is-ready"); $("#ysa-empty").hidden = true;
    fields.disabled = false; $("#ysa-prev").disabled = false; $("#ysa-next").disabled = false; $("#ysa-zoom").disabled = false; $("#ysa-timeline-zoom").disabled = false; $("#ysa-new").disabled = false; $("#ysa-export").disabled = false; $("#ysa-exclusion-toggle").disabled = false; $("#ysa-exclusion-reason").disabled = false; $("#ysa-scene-toggle").disabled = false; $("#ysa-scene-type").disabled = false; $("#ysa-serve-begin").disabled = false; $("#ysa-serve-end").disabled = false;
    $("#ysa-video-identity").textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
    clearEditor();
    $("#ysa-session-state").textContent = "新会话 · 将保存到 annotations/score_annotations";
  }
  try {
    const digest = await fingerprint(file);
    if (currentDocument && currentDocument.video.browser_identity === identity && currentDocument.video.fingerprint.value !== digest) {
      currentDocument.video.fingerprint.value = digest; persist();
    }
  } catch { /* File metadata remains sufficient when Web Crypto is unavailable. */ }
}

videoFile.addEventListener("change", () => openVideo(videoFile.files[0]));
video.addEventListener("loadedmetadata", () => {
  if (Number.isFinite(video.duration) && video.duration > 0) currentDocument.video.duration_s = roundTime(video.duration);
  currentDocument.video.width = video.videoWidth;
  currentDocument.video.height = video.videoHeight;
  persist(); renderTimeline(); updateTimecode(); clearEditor();
});
video.addEventListener("durationchange", () => {
  if (!currentDocument || !Number.isFinite(video.duration) || video.duration <= 0) return;
  const duration = roundTime(video.duration);
  if (currentDocument.video.duration_s === duration) return;
  currentDocument.video.duration_s = duration;
  persist(); renderTimeline(); updateTimecode();
});
video.addEventListener("timeupdate", updateTimecode);
video.addEventListener("seeked", updateTimecode);
$("#ysa-prev").addEventListener("click", () => { video.pause(); video.currentTime = Math.max(0, video.currentTime - 1 / fps()); });
$("#ysa-next").addEventListener("click", () => { video.pause(); video.currentTime = Math.min(mediaDuration() || Infinity, video.currentTime + 1 / fps()); });
$("#ysa-zoom").addEventListener("input", event => { const zoom = Number(event.target.value); video.style.width = `${zoom}%`; $("#ysa-zoom-value").textContent = `${zoom}%`; });
$("#ysa-timeline-zoom").addEventListener("input", event => {
  timelineScale = Number(event.target.value);
  $("#ysa-timeline-zoom-value").textContent = `${timelineScale}×`;
  renderTimeline();
});
element.querySelectorAll(".ysa__track-playhead").forEach(playhead => playhead.addEventListener("pointerdown", beginPlayheadDrag));
element.querySelectorAll(".ysa__track-lane").forEach(lane => lane.addEventListener("pointerdown", beginTrackDraft));
familyButtons.forEach(button => button.addEventListener("click", () => { setFamily(button.dataset.family); if (selectedEventId) editorDirty = true; }));
element.querySelectorAll("[data-set-time]").forEach(button => button.addEventListener("click", () => {
  if (button.dataset.setTime === "anchor") { setAnchorFromCurrent(); return; }
  $(`#ysa-${button.dataset.setTime}`).value = roundTime(video.currentTime || 0);
  if (selectedEventId) editorDirty = true;
}));
$("#ysa-use-anchor").addEventListener("click", setAnchorFromCurrent);
[$("#ysa-score"),$("#ysa-penalty"),$("#ysa-anchor"),$("#ysa-start"),$("#ysa-end"),$("#ysa-action")].forEach(node => {
  const markEditorDirty = () => {
    if (node.id === "ysa-anchor") {
      if (selectedEventId) {
        const selectedEvent = currentDocument?.events.find(event => event.event_id === selectedEventId);
        if (selectedEvent) selectedEvent.timing.anchor_source = "manual";
      } else if (pendingEventStart !== null) {
        pendingEventAnchor = roundTime(numeric(node));
      }
    }
    if (selectedEventId) editorDirty = true;
  };
  node.addEventListener("input", markEditorDirty);
  node.addEventListener("change", markEditorDirty);
});
$("#ysa-new").addEventListener("click", () => {
  if (selectedEventId && !syncSelectedFromEditor("事件已自动更新", false)) return;
  pendingExclusionStart = null;
  pendingSceneStart = null;
  clearEditor();
});
$("#ysa-save").addEventListener("click", () => {
  if (selectedEventId) return;
  const current = snapToFrame(video.currentTime || 0);
  if (pendingEventStart === null) {
    pendingEventStart = current;
    pendingEventAnchor = null;
    $("#ysa-start").value = current; $("#ysa-anchor").value = current; $("#ysa-end").value = current;
    $("#ysa-editor-title").textContent = "正在添加计分事件";
    $("#ysa-save").textContent = "结束并添加";
    return;
  }
  const start = Math.min(pendingEventStart, current);
  const end = Math.max(pendingEventStart, current);
  const anchor = pendingEventAnchor === null ? end : pendingEventAnchor;
  if (anchor < start || anchor > end) {
    $("#ysa-validation").textContent = "Anchor 必须位于最终 Evidence 区间内。";
    return;
  }
  $("#ysa-start").value = start; $("#ysa-anchor").value = anchor; $("#ysa-end").value = end;
  const event = buildEvent(); if (!event) return;
  currentDocument.events.push(event);
  $("#ysa-validation").textContent = ""; persist("事件已添加"); loadEvent(event.event_id, false);
});
function addServeReceiveMarker(kind) {
  if (!currentDocument) return;
  if (selectedEventId && !syncSelectedFromEditor("事件已自动更新", false)) return;
  const keyframe = snapToFrame(video.currentTime || 0);
  const padding = 0.4;
  const start = Math.max(0, roundTime(keyframe - padding));
  const end = Math.min(mediaDuration() || keyframe + padding, roundTime(keyframe + padding));
  currentDocument.serve_receive_events = currentDocument.serve_receive_events || [];
  const timestamp = nowIso();
  currentDocument.serve_receive_events.push({
    event_id:uid(), sequence_index:currentDocument.serve_receive_events.length + 1,
    marker:{track:"serve_receive", type:kind, display_name:kind},
    timing:{anchor_s:roundTime(keyframe), anchor_source:"manual", evidence_start_s:start, evidence_end_s:end, anchor_frame_index:frameAt(keyframe), evidence_start_frame_index:frameAt(start), evidence_end_frame_index:frameAt(end), fps_assumption:fps(), boundary_uncertainty_allowed:true},
    action_name:null, created_at:timestamp, updated_at:timestamp
  });
  persist(`${kind} 标注已添加`); renderEvents();
}
$("#ysa-serve-begin").addEventListener("click", () => addServeReceiveMarker("begin"));
$("#ysa-serve-end").addEventListener("click", () => addServeReceiveMarker("end"));
$("#ysa-exclusion-toggle").addEventListener("click", () => {
  if (!currentDocument) return;
  if (selectedServeReceiveId !== null && pendingExclusionStart === null && pendingSceneStart === null) {
    currentDocument.serve_receive_events = (currentDocument.serve_receive_events || []).filter(event => event.event_id !== selectedServeReceiveId);
    currentDocument.serve_receive_events.forEach((event, index) => event.sequence_index = index + 1);
    selectedServeReceiveId = null;
    updateExclusionControls(); persist("发球/收球区间已删除"); renderEvents();
    return;
  }
  if (selectedSceneId !== null && pendingExclusionStart === null && pendingSceneStart === null) {
    currentDocument.scene_intervals = currentDocument.scene_intervals.filter(interval => interval.scene_interval_id !== selectedSceneId);
    selectedSceneId = null;
    updateExclusionControls(); updateSceneControls(); persist("场景区间已删除"); renderEvents();
    return;
  }
  if (selectedExclusionId !== null && pendingExclusionStart === null) {
    currentDocument.excluded_intervals = currentDocument.excluded_intervals.filter(interval => interval.exclusion_id !== selectedExclusionId);
    selectedExclusionId = null;
    updateExclusionControls(); persist("不可标记区间已删除"); renderEvents();
    return;
  }
  if (selectedEventId && !syncSelectedFromEditor("事件已自动更新", false)) return;
  clearEditor(false);
  const current = snapToFrame(video.currentTime || 0);
  if (pendingExclusionStart === null) {
    pendingExclusionStart = current;
    updateExclusionControls(); updateSceneControls();
    toast("已记录不可标记区间起点");
    return;
  }
  addExclusion(pendingExclusionStart, current);
});
$("#ysa-scene-toggle").addEventListener("click", () => {
  if (!currentDocument) return;
  if (selectedEventId && !syncSelectedFromEditor("事件已自动更新", false)) return;
  clearEditor(false);
  const current = snapToFrame(video.currentTime || 0);
  if (pendingSceneStart === null) {
    pendingSceneStart = current;
    updateSceneControls(); updateExclusionControls();
    toast("已记录场景区间起点");
    return;
  }
  addSceneInterval(pendingSceneStart, current);
});
$("#ysa-delete").addEventListener("click", () => {
  if (!selectedEventId) return;
  currentDocument.events = currentDocument.events.filter(event => event.event_id !== selectedEventId);
  persist("事件已删除"); clearEditor();
});
element.addEventListener("click", clickEvent => {
  if (!selectedEventId) return;
  if (clickEvent.target.closest(".ysa__editor,.ysa__clip,#ysa-event-list tr[data-event-id],.ysa__player-panel,.ysa__timeline-heading,.ysa__track-playhead")) return;
  if (!syncSelectedFromEditor("事件已自动更新", false)) return;
  clearEditor(false);
  renderEvents();
});
[division,judge,fpsInput].forEach(node => node.addEventListener("change", () => {
  if (currentDocument) {
    currentDocument.events.forEach(event => {
      event.timing.fps_assumption = fps();
      event.timing.anchor_frame_index = frameAt(event.timing.anchor_s);
      event.timing.evidence_start_frame_index = frameAt(event.timing.evidence_start_s);
      event.timing.evidence_end_frame_index = frameAt(event.timing.evidence_end_s);
    });
    currentDocument.excluded_intervals.forEach(interval => {
      interval.fps_assumption = fps();
      interval.start_frame_index = frameAt(interval.start_s);
      interval.end_frame_index = frameAt(interval.end_s);
    });
    currentDocument.scene_intervals.forEach(interval => {
      interval.fps_assumption = fps();
      interval.start_frame_index = frameAt(interval.start_s);
      interval.end_frame_index = frameAt(interval.end_s);
    });
    persist("会话信息已更新"); renderEvents(); updateTimecode();
  }
}));
judge.addEventListener("blur", () => { if (!judge.value.trim()) judge.value = "judge1"; if (currentDocument) persist(); });
metadataFile.addEventListener("change", async () => {
  const file = metadataFile.files[0]; if (!file) return;
  try {
    const incoming = JSON.parse(await file.text());
    if (incoming.schema_version !== "yoyo_score_annotation_v2" || !Array.isArray(incoming.events) || !Array.isArray(incoming.scene_intervals)) throw new Error("schema");
    if (!currentDocument) { toast("请先导入对应视频"); metadataFile.value = ""; return; }
    const sameVideo = incoming.video?.browser_identity === currentDocument.video.browser_identity || (incoming.video?.fingerprint?.value && incoming.video.fingerprint.value === currentDocument.video.fingerprint?.value);
    if (!sameVideo && !window.confirm("元数据的视频身份与当前视频不同，仍要载入吗？")) return;
    currentDocument = normalizeDocument(incoming);
    currentDocument.video.browser_identity = videoIdentity(videoFile.files[0]);
    division.value = currentDocument.competition?.division || "1A"; judge.value = currentDocument.annotator?.judge || "judge1"; fpsInput.value = currentDocument.timing_basis?.fps_assumption || 30;
    persist("元数据已载入并保存"); clearEditor();
  } catch { toast("无法读取该元数据文件"); }
  metadataFile.value = "";
});
$("#ysa-export").addEventListener("click", async () => {
  if (!currentDocument) return;
  if (!syncSelectedFromEditor("事件已自动更新", false)) return;
  await saveQueue;
  downloadAnnotationDocument(currentDocument);
  toast("元数据已导出");
});
$("#ysa-manager-open").addEventListener("click", openSessionManager);
$("#ysa-manager-close").addEventListener("click", closeSessionManager);
$("#ysa-manager-backdrop").addEventListener("click", closeSessionManager);
element.addEventListener("keydown", event => {
  if (event.code === "Escape" && $("#ysa-manager").classList.contains("is-open")) { closeSessionManager(); return; }
  if (event.target.matches("input,select,textarea")) return;
  if (event.code === "Space") { event.preventDefault(); video.paused ? video.play() : video.pause(); }
  if (event.code === "ArrowLeft") { event.preventDefault(); $("#ysa-prev").click(); }
  if (event.code === "ArrowRight") { event.preventDefault(); $("#ysa-next").click(); }
});
setFamily("positive"); updateTimecode(); refreshSessionManager();
"""


def score_annotation_component_kwargs() -> dict[str, Any]:
    """Return the Gradio HTML arguments without importing Gradio here."""
    return {
        "value": SCORE_ANNOTATION_HTML,
        "css_template": SCORE_ANNOTATION_CSS,
        "js_on_load": SCORE_ANNOTATION_JS,
        "apply_default_css": False,
        "container": False,
        "padding": False,
        "server_functions": [
            save_score_annotation,
            load_score_annotation,
            load_score_annotation_session,
            resolve_score_video_source,
            list_score_annotations,
            delete_score_annotation,
        ],
    }


def score_annotation_schema() -> str:
    """Expose a compact machine-readable contract for documentation/tests."""
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "divisions": list(DIVISIONS),
            "score_ranges": {"positive": [0, 10], "negative": [-10, -1]},
            "major_penalties": MAJOR_PENALTIES,
            "anchor_sources": list(ANCHOR_SOURCES),
            "timing": ["anchor_s", "anchor_source", "evidence_start_s", "evidence_end_s"],
            "scene_intervals": {
                "fields": ["start_s", "end_s", "scene_type"],
                "scene_types": list(SCENE_TYPES),
            },
            "excluded_intervals": {
                "fields": ["start_s", "end_s", "reason", "training_eligible"],
                "reasons": list(EXCLUSION_REASONS),
                "training_eligible": False,
            },
            "serve_receive_events": {
                "fields": ["event_id", "sequence_index", "marker", "timing", "action_name", "created_at", "updated_at"],
                "kinds": ["begin", "end"],
                "window_s": 0.4,
            },
            "optional_fields": ["action_name"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
