"""Browser-based score-event annotation workbench and metadata contract."""

from __future__ import annotations

import json
from typing import Any


SCHEMA_VERSION = "yoyo_score_annotation_v1"
DIVISIONS = ("1A", "2A", "3A", "4A", "5A")
MAJOR_PENALTIES = {
    "restart": {"display_name": "重启", "score_delta": -1},
    "discard": {"display_name": "弃用", "score_delta": -3},
    "disassembly": {"display_name": "解体", "score_delta": -5},
}


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


SCORE_ANNOTATION_HTML = r"""
<div class="ysa" data-yoyo-score-annotation>
  <header class="ysa__header">
    <div>
      <h2>悠悠球计分标注</h2>
      <p id="ysa-session-state">选择本地视频开始新的标注会话</p>
    </div>
    <div class="ysa__header-actions">
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
          <button type="button" class="ysa__button ysa__button--primary" id="ysa-save">添加事件</button>
          <button type="button" class="ysa__button ysa__button--danger" id="ysa-delete" hidden>删除</button>
        </div>
      </fieldset>
    </aside>
  </main>

  <section class="ysa__timeline-panel" aria-label="三轨计分时间轴">
    <div class="ysa__timeline-heading">
      <h3>分轨时间轴</h3>
      <label>时间轴缩放
        <input id="ysa-timeline-zoom" type="range" min="1" max="8" value="1" step="0.5" disabled>
        <output id="ysa-timeline-zoom-value">1×</output>
      </label>
    </div>
    <div class="ysa__timeline-scroll" id="ysa-timeline-scroll">
      <div class="ysa__timeline-content" id="ysa-timeline-content">
        <div class="ysa__ruler"><span class="ysa__track-label">时间</span><div class="ysa__ruler-lane" id="ysa-ruler-lane"></div></div>
        <div class="ysa__track-row" data-family="positive"><span class="ysa__track-label">正向计分</span><div class="ysa__track-lane" data-track="positive" role="group" aria-label="正向计分轨"><div class="ysa__track-playhead"></div></div></div>
        <div class="ysa__track-row" data-family="negative"><span class="ysa__track-label">负向计分</span><div class="ysa__track-lane" data-track="negative" role="group" aria-label="负向计分轨"><div class="ysa__track-playhead"></div></div></div>
        <div class="ysa__track-row" data-family="major_penalty"><span class="ysa__track-label">重点扣分</span><div class="ysa__track-lane" data-track="major_penalty" role="group" aria-label="重点扣分轨"><div class="ysa__track-playhead"></div></div></div>
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
.ysa__timeline-heading { align-items:center; display:flex; justify-content:space-between; margin:0 8px 10px; }
.ysa__timeline-heading label { align-items:center; display:flex; flex-direction:row; gap:8px; margin:0; }
.ysa__timeline-heading input { accent-color:var(--accent); height:auto; padding:0; width:150px; }
.ysa__timeline-heading output { color:var(--muted); min-width:28px; }
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
.ysa__track-playhead { background:#e29c35; bottom:0; left:0; pointer-events:none; position:absolute; top:0; width:2px; z-index:7; }
.ysa__clip { align-items:center; background:#287c65; border:1px solid #60aa93; border-radius:3px; bottom:7px; color:#fff; cursor:grab; display:flex; font-size:10px; font-weight:700; min-width:8px; overflow:visible; padding:0 7px; position:absolute; text-overflow:ellipsis; top:7px; touch-action:none; white-space:nowrap; z-index:3; }
.ysa__clip[data-family="negative"] { background:#a84a4a; border-color:#d28585; }
.ysa__clip[data-family="major_penalty"] { background:#8b5d29; border-color:#c49459; }
.ysa__clip.is-selected { box-shadow:0 0 0 2px #f2c56e; z-index:5; }
.ysa__clip-anchor { background:#ffe4a6; bottom:-3px; cursor:ew-resize; position:absolute; top:-3px; width:3px; z-index:4; }
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
const storagePrefix = "yoyo-score-annotation:v1:";
const penalties = {restart:{name:"重启",delta:-1},discard:{name:"弃用",delta:-3},disassembly:{name:"解体",delta:-5}};
let objectUrl = null;
let currentDocument = null;
let currentKey = null;
let selectedEventId = null;
let currentFamily = "positive";
let timelineScale = 1;
let toastTimer = null;

const nowIso = () => new Date().toISOString();
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const numeric = (node) => Number.parseFloat(node.value || "0");
const fps = () => clamp(numeric(fpsInput) || 30, 1, 240);
const roundTime = (value) => Math.round(value * 1000) / 1000;
const snapToFrame = (value) => roundTime(Math.round(value * fps()) / fps());
const frameAt = (value) => Math.max(0, Math.round(value * fps()));
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
const storageKey = (identity) => `${storagePrefix}${identity}`;
const eventName = (event) => event.label.display_name;

function blankDocument(file) {
  const timestamp = nowIso();
  return {
    schema_version:"yoyo_score_annotation_v1",
    annotation_id:uid(),
    revision:0,
    video:{
      video_id:uid(), file_name:file.name, file_size_bytes:file.size,
      last_modified_ms:file.lastModified, browser_identity:videoIdentity(file),
      duration_s:null, width:null, height:null,
      fingerprint:{algorithm:"sha256:first-last-1m+size", value:null}
    },
    competition:{division:division.value || "1A"},
    annotator:{judge:(judge.value || "judge1").trim() || "judge1"},
    timing_basis:{unit:"seconds", fps_assumption:fps(), frame_index_rounding:"nearest"},
    events:[], created_at:timestamp, updated_at:timestamp
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
  if (!currentDocument || !currentKey) return;
  currentDocument.revision = Number(currentDocument.revision || 0) + 1;
  currentDocument.updated_at = nowIso();
  currentDocument.competition.division = division.value;
  currentDocument.annotator.judge = (judge.value || "judge1").trim() || "judge1";
  currentDocument.timing_basis.fps_assumption = fps();
  localStorage.setItem(currentKey, JSON.stringify(currentDocument));
  $("#ysa-session-state").textContent = `已自动保存 · 修订 ${currentDocument.revision} · ${new Date(currentDocument.updated_at).toLocaleTimeString()}`;
  if (message) toast(message);
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

function clearEditor() {
  selectedEventId = null;
  $("#ysa-editor-title").textContent = "新建计分事件";
  $("#ysa-save").textContent = "添加事件";
  $("#ysa-delete").hidden = true;
  $("#ysa-action").value = "";
  const t = roundTime(video.currentTime || 0);
  $("#ysa-anchor").value = t; $("#ysa-start").value = t; $("#ysa-end").value = t;
  $("#ysa-validation").textContent = "";
  renderEvents();
}

function populateEditor(event) {
  setFamily(event.label.family);
  $("#ysa-score").value = event.label.score_delta;
  if (event.label.penalty_type) $("#ysa-penalty").value = event.label.penalty_type;
  $("#ysa-anchor").value = event.timing.anchor_s;
  $("#ysa-start").value = event.timing.evidence_start_s;
  $("#ysa-end").value = event.timing.evidence_end_s;
  $("#ysa-action").value = event.action_name || "";
  $("#ysa-editor-title").textContent = "编辑计分事件";
  $("#ysa-save").textContent = "保存修改";
  $("#ysa-delete").hidden = false;
}

function loadEvent(id, seekToAnchor = true) {
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

function buildEvent() {
  const start = roundTime(numeric($("#ysa-start")));
  const anchor = roundTime(numeric($("#ysa-anchor")));
  const end = roundTime(numeric($("#ysa-end")));
  if (start < 0 || anchor < 0 || end < 0 || start > anchor || anchor > end) {
    $("#ysa-validation").textContent = "时间必须满足 Evidence 起点 ≤ Anchor ≤ Evidence 终点。";
    return null;
  }
  if (video.duration && end > video.duration + .001) {
    $("#ysa-validation").textContent = "Evidence 终点不能超过视频时长。";
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
  const timestamp = nowIso();
  return {
    event_id:existing?.event_id || uid(), sequence_index:0,
    label:{family:currentFamily, code:currentFamily === "major_penalty" ? `major_penalty_${penaltyType}` : `score_${currentFamily}`, display_name:displayName, score_delta:scoreDelta, penalty_type:penaltyType},
    timing:{anchor_s:anchor, evidence_start_s:start, evidence_end_s:end, anchor_frame_index:frameAt(anchor), evidence_start_frame_index:frameAt(start), evidence_end_frame_index:frameAt(end), fps_assumption:fps(), boundary_uncertainty_allowed:true},
    action_name:$("#ysa-action").value.trim() || null,
    created_at:existing?.created_at || timestamp, updated_at:timestamp
  };
}

function syncSelectedFromEditor(message = "手动标注已同步") {
  if (!selectedEventId || !currentDocument) return;
  const event = buildEvent();
  if (!event) return;
  const index = currentDocument.events.findIndex(item => item.event_id === selectedEventId);
  if (index < 0) return;
  currentDocument.events[index] = event;
  persist(message);
  renderEvents();
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
  const duration = video.duration || currentDocument?.video.duration_s || 0;
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
    clip.innerHTML = `<span>${event.label.score_delta > 0 ? "+" : ""}${event.label.score_delta}</span><i class="ysa__clip-handle" data-handle="start"></i><i class="ysa__clip-anchor" data-handle="anchor"></i><i class="ysa__clip-handle" data-handle="end"></i>`;
    updateClipGeometry(clip, event, duration);
    clip.addEventListener("pointerdown", pointerEvent => beginClipDrag(pointerEvent, event, clip));
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
  clip.querySelector(".ysa__clip-anchor").style.left = `calc(${clamp(anchorRatio, 0, 100)}% - 1px)`;
}

function beginClipDrag(pointerEvent, scoreEvent, clip) {
  pointerEvent.preventDefault(); pointerEvent.stopPropagation();
  const mode = pointerEvent.target.dataset.handle || "move";
  const lane = clip.parentElement;
  const duration = video.duration || currentDocument.video.duration_s;
  const origin = {...scoreEvent.timing};
  const startX = pointerEvent.clientX;
  let moved = false;
  selectedEventId = scoreEvent.event_id;
  populateEditor(scoreEvent);
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
    if (moved) { persist("轨道事件已更新"); renderEvents(); }
    else loadEvent(scoreEvent.event_id, true);
  };
  clip.addEventListener("pointermove", onMove); clip.addEventListener("pointerup", onUp); clip.addEventListener("pointercancel", onUp);
}

function beginTrackDraft(pointerEvent) {
  const lane = pointerEvent.currentTarget;
  if (pointerEvent.target !== lane || !currentDocument || !video.duration) return;
  pointerEvent.preventDefault();
  const rect = lane.getBoundingClientRect();
  const at = clientX => clamp((clientX - rect.left) / rect.width, 0, 1) * video.duration;
  const startTime = at(pointerEvent.clientX);
  const startX = pointerEvent.clientX;
  const draft = document.createElement("div");
  draft.className = "ysa__draft-clip"; lane.append(draft);
  lane.setPointerCapture(pointerEvent.pointerId);
  const onMove = moveEvent => {
    const current = at(moveEvent.clientX);
    draft.style.left = `${100 * Math.min(startTime, current) / video.duration}%`;
    draft.style.width = `${100 * Math.abs(current - startTime) / video.duration}%`;
  };
  const onUp = upEvent => {
    lane.removeEventListener("pointermove", onMove); lane.removeEventListener("pointerup", onUp); lane.removeEventListener("pointercancel", onUp); draft.remove();
    const endTime = at(upEvent.clientX);
    if (Math.abs(upEvent.clientX - startX) < 4) { video.currentTime = endTime; return; }
    selectedEventId = null;
    setFamily(lane.dataset.track);
    if (lane.dataset.track === "positive") $("#ysa-score").value = "1";
    if (lane.dataset.track === "negative") $("#ysa-score").value = "-1";
    if (lane.dataset.track === "major_penalty") $("#ysa-penalty").value = "restart";
    const start = snapToFrame(Math.min(startTime, endTime));
    const end = snapToFrame(Math.max(startTime, endTime));
    $("#ysa-start").value = start; $("#ysa-anchor").value = end; $("#ysa-end").value = end; $("#ysa-action").value = "";
    const event = buildEvent();
    if (event) { currentDocument.events.push(event); persist("已从轨道添加事件"); loadEvent(event.event_id, true); }
  };
  lane.addEventListener("pointermove", onMove); lane.addEventListener("pointerup", onUp); lane.addEventListener("pointercancel", onUp);
}

function updateTimecode() {
  const time = video.currentTime || 0;
  $("#ysa-timecode").textContent = `${formatTime(time)} · f${frameAt(time)}`;
  const ratio = video.duration ? clamp(time / video.duration, 0, 1) : 0;
  element.querySelectorAll(".ysa__track-playhead").forEach(node => node.style.left = `${ratio * 100}%`);
}

async function openVideo(file) {
  if (!file) return;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  currentKey = storageKey(videoIdentity(file));
  const stored = localStorage.getItem(currentKey);
  try { currentDocument = stored ? JSON.parse(stored) : blankDocument(file); }
  catch { currentDocument = blankDocument(file); }
  if (currentDocument.schema_version !== "yoyo_score_annotation_v1") currentDocument = blankDocument(file);
  division.value = currentDocument.competition?.division || "1A";
  judge.value = currentDocument.annotator?.judge || "judge1";
  fpsInput.value = currentDocument.timing_basis?.fps_assumption || 30;
  video.src = objectUrl;
  video.classList.add("is-ready"); $("#ysa-empty").hidden = true;
  fields.disabled = false; $("#ysa-prev").disabled = false; $("#ysa-next").disabled = false; $("#ysa-zoom").disabled = false; $("#ysa-timeline-zoom").disabled = false; $("#ysa-new").disabled = false; $("#ysa-export").disabled = false;
  $("#ysa-video-identity").textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
  clearEditor();
  $("#ysa-session-state").textContent = stored ? `已恢复 ${currentDocument.events.length} 个事件 · 修订 ${currentDocument.revision}` : "新会话 · 修改将自动保存";
  try {
    const digest = await fingerprint(file);
    if (currentDocument && currentDocument.video.browser_identity === videoIdentity(file) && currentDocument.video.fingerprint.value !== digest) {
      currentDocument.video.fingerprint.value = digest; persist();
    }
  } catch { /* File metadata remains sufficient when Web Crypto is unavailable. */ }
}

videoFile.addEventListener("change", () => openVideo(videoFile.files[0]));
video.addEventListener("loadedmetadata", () => {
  currentDocument.video.duration_s = roundTime(video.duration);
  currentDocument.video.width = video.videoWidth;
  currentDocument.video.height = video.videoHeight;
  persist(); renderTimeline(); updateTimecode(); clearEditor();
});
video.addEventListener("timeupdate", updateTimecode);
video.addEventListener("seeked", updateTimecode);
$("#ysa-prev").addEventListener("click", () => { video.pause(); video.currentTime = Math.max(0, video.currentTime - 1 / fps()); });
$("#ysa-next").addEventListener("click", () => { video.pause(); video.currentTime = Math.min(video.duration || Infinity, video.currentTime + 1 / fps()); });
$("#ysa-zoom").addEventListener("input", event => { const zoom = Number(event.target.value); video.style.width = `${zoom}%`; $("#ysa-zoom-value").textContent = `${zoom}%`; });
$("#ysa-timeline-zoom").addEventListener("input", event => {
  timelineScale = Number(event.target.value);
  $("#ysa-timeline-zoom-value").textContent = `${timelineScale}×`;
  renderTimeline();
});
element.querySelectorAll(".ysa__track-lane").forEach(lane => lane.addEventListener("pointerdown", beginTrackDraft));
familyButtons.forEach(button => button.addEventListener("click", () => { setFamily(button.dataset.family); syncSelectedFromEditor(); }));
element.querySelectorAll("[data-set-time]").forEach(button => button.addEventListener("click", () => {
  $(`#ysa-${button.dataset.setTime}`).value = roundTime(video.currentTime || 0);
  syncSelectedFromEditor();
}));
$("#ysa-use-anchor").addEventListener("click", () => { $("#ysa-anchor").value = roundTime(video.currentTime || 0); syncSelectedFromEditor(); });
[$("#ysa-score"),$("#ysa-penalty"),$("#ysa-anchor"),$("#ysa-start"),$("#ysa-end"),$("#ysa-action")].forEach(node => node.addEventListener("change", () => syncSelectedFromEditor()));
$("#ysa-new").addEventListener("click", clearEditor);
$("#ysa-save").addEventListener("click", () => {
  const event = buildEvent(); if (!event) return;
  const index = currentDocument.events.findIndex(item => item.event_id === event.event_id);
  if (index >= 0) currentDocument.events[index] = event; else currentDocument.events.push(event);
  $("#ysa-validation").textContent = ""; persist(index >= 0 ? "事件已更新" : "事件已添加"); clearEditor();
});
$("#ysa-delete").addEventListener("click", () => {
  if (!selectedEventId) return;
  currentDocument.events = currentDocument.events.filter(event => event.event_id !== selectedEventId);
  persist("事件已删除"); clearEditor();
});
[division,judge,fpsInput].forEach(node => node.addEventListener("change", () => {
  if (currentDocument) {
    currentDocument.events.forEach(event => {
      event.timing.fps_assumption = fps();
      event.timing.anchor_frame_index = frameAt(event.timing.anchor_s);
      event.timing.evidence_start_frame_index = frameAt(event.timing.evidence_start_s);
      event.timing.evidence_end_frame_index = frameAt(event.timing.evidence_end_s);
    });
    persist("会话信息已更新"); renderEvents(); updateTimecode();
  }
}));
judge.addEventListener("blur", () => { if (!judge.value.trim()) judge.value = "judge1"; if (currentDocument) persist(); });
metadataFile.addEventListener("change", async () => {
  const file = metadataFile.files[0]; if (!file) return;
  try {
    const incoming = JSON.parse(await file.text());
    if (incoming.schema_version !== "yoyo_score_annotation_v1" || !Array.isArray(incoming.events)) throw new Error("schema");
    if (!currentDocument) { toast("请先导入对应视频"); metadataFile.value = ""; return; }
    const sameVideo = incoming.video?.browser_identity === currentDocument.video.browser_identity || (incoming.video?.fingerprint?.value && incoming.video.fingerprint.value === currentDocument.video.fingerprint?.value);
    if (!sameVideo && !window.confirm("元数据的视频身份与当前视频不同，仍要载入吗？")) return;
    currentDocument = incoming;
    currentDocument.video.browser_identity = videoIdentity(videoFile.files[0]);
    division.value = currentDocument.competition?.division || "1A"; judge.value = currentDocument.annotator?.judge || "judge1"; fpsInput.value = currentDocument.timing_basis?.fps_assumption || 30;
    persist("元数据已载入并保存"); clearEditor();
  } catch { toast("无法读取该元数据文件"); }
  metadataFile.value = "";
});
$("#ysa-export").addEventListener("click", () => {
  if (!currentDocument) return;
  persist();
  const stem = currentDocument.video.file_name.replace(/\.[^.]+$/, "").replace(/[^\w\-\u4e00-\u9fff]+/g, "_");
  const blob = new Blob([JSON.stringify(currentDocument, null, 2) + "\n"], {type:"application/json"});
  const url = URL.createObjectURL(blob); const link = document.createElement("a");
  link.href = url; link.download = `${stem}.score-annotation.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); toast("元数据已导出");
});
element.addEventListener("keydown", event => {
  if (event.target.matches("input,select,textarea")) return;
  if (event.code === "Space") { event.preventDefault(); video.paused ? video.play() : video.pause(); }
  if (event.code === "ArrowLeft") { event.preventDefault(); $("#ysa-prev").click(); }
  if (event.code === "ArrowRight") { event.preventDefault(); $("#ysa-next").click(); }
});
setFamily("positive"); updateTimecode();
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
    }


def score_annotation_schema() -> str:
    """Expose a compact machine-readable contract for documentation/tests."""
    return json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "divisions": list(DIVISIONS),
            "score_ranges": {"positive": [0, 10], "negative": [-10, -1]},
            "major_penalties": MAJOR_PENALTIES,
            "timing": ["anchor_s", "evidence_start_s", "evidence_end_s"],
            "optional_fields": ["action_name"],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
