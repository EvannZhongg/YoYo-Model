"""Consecutive-frame annotation workflow built on the canonical editor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.files import atomic_write_text
from workbench import dataset_annotation as base


CONSECUTIVE_FILENAME = "consecutive_groups.json"
CONSECUTIVE_SCHEMA_VERSION = "yoyo_consecutive_groups_v1"


def _read_groups(dataset_path: str | Path) -> tuple[Path, dict[str, Any]]:
    dataset = base._managed_dataset_path(dataset_path)
    path = dataset / CONSECUTIVE_FILENAME
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid consecutive metadata: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"consecutive metadata must be an object: {path}")
    if document.get("schema_version") != CONSECUTIVE_SCHEMA_VERSION:
        raise ValueError(f"unsupported consecutive group schema: {path}")
    if document.get("dataset_id") != dataset.name:
        raise ValueError("consecutive metadata dataset_id does not match its directory")
    groups = document.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("consecutive metadata must contain at least one group")

    _, labels_root, _, = base._annotation_roots(dataset)
    seen_groups: set[str] = set()
    seen_samples: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("every consecutive group must be an object")
        group_id = str(group.get("group_id") or "")
        frames = group.get("frames")
        if not group_id or group_id in seen_groups:
            raise ValueError("consecutive group_id values must be non-empty and unique")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"consecutive group has no frames: {group_id}")
        seen_groups.add(group_id)
        indices: list[int] = []
        keys: list[str] = []
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError(f"invalid frame entry in consecutive group: {group_id}")
            key = Path(str(frame.get("sample_key") or ""))
            if key.is_absolute() or ".." in key.parts or key.suffix.lower() != ".json":
                raise ValueError(f"invalid sample key in consecutive group: {key}")
            label_path = (labels_root / key).resolve()
            if not label_path.is_relative_to(labels_root.resolve()) or not label_path.is_file():
                raise ValueError(f"consecutive group points to a missing label: {key.as_posix()}")
            normalized_key = key.as_posix()
            if normalized_key in seen_samples:
                raise ValueError(f"sample belongs to more than one consecutive group: {normalized_key}")
            seen_samples.add(normalized_key)
            keys.append(normalized_key)
            indices.append(int(frame["frame_index"]))
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(f"frame indices are not consecutive in group: {group_id}")
        if int(group.get("selected_start_frame")) != indices[0]:
            raise ValueError(f"selected_start_frame disagrees with group frames: {group_id}")
        if int(group.get("selected_end_frame")) != indices[-1]:
            raise ValueError(f"selected_end_frame disagrees with group frames: {group_id}")
        if str(group.get("start_sample_key") or "") != keys[0]:
            raise ValueError(f"start_sample_key disagrees with group frames: {group_id}")
    return dataset, document


def _write_groups(dataset: Path, document: dict[str, Any]) -> None:
    path = dataset / CONSECUTIVE_FILENAME
    payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    atomic_write_text(path, payload)


def list_consecutive_annotation_datasets() -> list[dict[str, str]]:
    """List only datasets carrying the new consecutive-frame mapping."""
    results = []
    for item in base.list_annotation_datasets(include_consecutive=True):
        path = Path(item["path"])
        if (path / CONSECUTIVE_FILENAME).is_file():
            try:
                _read_groups(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            results.append(item)
    return results


def _group_summary(group: dict[str, Any]) -> dict[str, Any]:
    frames = group["frames"]
    return {
        "group_id": str(group["group_id"]),
        "name": str(group.get("source_group") or group["group_id"]),
        "source_video": str(group.get("source_video") or ""),
        "original_start_frame": int(group.get("original_start_frame", frames[0]["frame_index"])),
        "original_end_frame": int(group.get("original_end_frame", frames[-1]["frame_index"])),
        "selected_start_frame": int(group["selected_start_frame"]),
        "selected_end_frame": int(group["selected_end_frame"]),
        "frame_count": len(frames),
        "start_sample_key": str(group["start_sample_key"]),
        "propagated": bool(group.get("propagated_from_start_at_utc")),
    }


def open_consecutive_annotation_dataset(dataset_path: str) -> dict[str, Any]:
    dataset, metadata = _read_groups(dataset_path)
    opened = base.open_annotation_dataset(str(dataset))
    summaries = {sample["key"]: sample for sample in opened["samples"]}
    ordered_samples: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for group in metadata["groups"]:
        groups.append(_group_summary(group))
        for position, frame in enumerate(group["frames"]):
            key = str(frame["sample_key"])
            if key not in summaries:
                raise ValueError(f"consecutive sample could not be opened: {key}")
            ordered_samples.append({
                **summaries[key],
                "group_id": str(group["group_id"]),
                "group_position": position,
                "is_start_frame": position == 0,
            })
    return {
        **opened,
        "sample_count": len(ordered_samples),
        "reviewed_count": sum(1 for sample in ordered_samples if sample["reviewed"]),
        "samples": ordered_samples,
        "groups": groups,
        "metadata_path": str(dataset / CONSECUTIVE_FILENAME),
    }


def select_consecutive_group_range(
    dataset_path: str,
    group_id: str,
    start_frame: int,
    end_frame: int,
) -> dict[str, Any]:
    """Trim one group mapping to an inclusive range without deleting source files."""
    dataset, document = _read_groups(dataset_path)
    group = next(
        (item for item in document["groups"] if str(item["group_id"]) == str(group_id)),
        None,
    )
    if group is None:
        raise ValueError("consecutive group was not found")
    start_frame, end_frame = int(start_frame), int(end_frame)
    if start_frame > end_frame:
        raise ValueError("start frame must not be after end frame")
    selected = [
        frame for frame in group["frames"]
        if start_frame <= int(frame["frame_index"]) <= end_frame
    ]
    if not selected or int(selected[0]["frame_index"]) != start_frame:
        raise ValueError("start frame is not available in this group")
    if int(selected[-1]["frame_index"]) != end_frame:
        raise ValueError("end frame is not available in this group")
    group["frames"] = selected
    group["selected_start_frame"] = start_frame
    group["selected_end_frame"] = end_frame
    group["start_sample_key"] = str(selected[0]["sample_key"])
    group["propagated_from_start_at_utc"] = None
    group["range_selected_at_utc"] = base._utc_now()
    document["updated_at_utc"] = base._utc_now()
    with base._STORAGE_LOCK:
        _write_groups(dataset, document)
    return open_consecutive_annotation_dataset(str(dataset))


def _copyable_edit(annotation: dict[str, Any], reviewer: str) -> dict[str, Any]:
    return {
        "yoyo_visibility": annotation.get("visibility"),
        "yoyo_not_visible_reason": annotation.get("yoyo_not_visible_reason"),
        "trick_orientation": annotation.get("trick_orientation"),
        "yoyo_bbox_pixel": annotation.get("yoyo_bbox_pixel"),
        "string_visibility": annotation.get("string_visibility"),
        "string_polylines_pixel": annotation.get("string_polylines_pixel") or [],
        "string_review_status": annotation.get("string_review_status"),
        "bbox_review_status": annotation.get("bbox_review_status"),
        "reviewer": reviewer,
        "notes": annotation.get("notes") or "",
    }


def save_consecutive_annotation_sample(
    dataset_path: str,
    sample_key: str,
    edit_json: str | dict[str, Any],
    propagate_remaining: bool = False,
) -> dict[str, Any]:
    """Save one frame and optionally overwrite every later frame in its group."""
    dataset, metadata = _read_groups(dataset_path)
    edit = json.loads(edit_json) if isinstance(edit_json, str) else edit_json
    if not isinstance(edit, dict):
        raise ValueError("annotation edit must be an object")
    group = next(
        (
            item for item in metadata["groups"]
            if any(str(frame["sample_key"]) == sample_key for frame in item["frames"])
        ),
        None,
    )
    if group is None:
        raise ValueError("sample is outside the active consecutive group mapping")

    result = base.save_annotation_sample(str(dataset), sample_key, edit)
    propagated: list[str] = []
    if propagate_remaining:
        reviewer = str(edit.get("reviewer") or "workbench-reviewer")
        copied_edit = _copyable_edit(result["annotation"], reviewer)
        source_position = next(
            index for index, frame in enumerate(group["frames"])
            if str(frame["sample_key"]) == sample_key
        )
        for frame in group["frames"][source_position + 1:]:
            target_key = str(frame["sample_key"])
            base.save_annotation_sample(str(dataset), target_key, copied_edit)
            propagated.append(target_key)
        group["propagated_from_start_at_utc"] = base._utc_now()
        group["propagated_from_sample_key"] = sample_key
        group["propagated_sample_count"] = len(propagated)
        group.setdefault("propagation_history", []).append({
            "created_at_utc": group["propagated_from_start_at_utc"],
            "source_sample_key": sample_key,
            "propagated_sample_count": len(propagated),
        })
        metadata["updated_at_utc"] = base._utc_now()
        with base._STORAGE_LOCK:
            _write_groups(dataset, metadata)
    return {
        **result,
        "propagated_count": len(propagated),
        "propagated_sample_keys": propagated,
    }


def ui_list_consecutive_annotation_datasets(_payload: object = None) -> list[dict[str, str]]:
    return list_consecutive_annotation_datasets()


def ui_open_consecutive_annotation_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    return open_consecutive_annotation_dataset(str((payload or {}).get("dataset_path") or ""))


def ui_load_consecutive_annotation_sample(payload: dict[str, Any]) -> dict[str, Any]:
    return base.load_annotation_sample(
        str((payload or {}).get("dataset_path") or ""),
        str((payload or {}).get("sample_key") or ""),
    )


def ui_save_consecutive_annotation_sample(payload: dict[str, Any]) -> dict[str, Any]:
    return save_consecutive_annotation_sample(
        str((payload or {}).get("dataset_path") or ""),
        str((payload or {}).get("sample_key") or ""),
        (payload or {}).get("edit") or {},
        bool((payload or {}).get("propagate_remaining", False)),
    )


def ui_set_consecutive_annotation_sample_reviewed(payload: dict[str, Any]) -> dict[str, Any]:
    return base.set_annotation_sample_reviewed(
        str((payload or {}).get("dataset_path") or ""),
        str((payload or {}).get("sample_key") or ""),
        str((payload or {}).get("reviewer") or ""),
        bool((payload or {}).get("confirmed", True)),
    )


def ui_select_consecutive_group_range(payload: dict[str, Any]) -> dict[str, Any]:
    return select_consecutive_group_range(
        str((payload or {}).get("dataset_path") or ""),
        str((payload or {}).get("group_id") or ""),
        int((payload or {}).get("start_frame")),
        int((payload or {}).get("end_frame")),
    )


def _component_html() -> str:
    html = base.DATASET_ANNOTATION_HTML.replace("yda", "yca")
    html = html.replace("数据标注</h2>", "连续帧标注</h2>", 1)
    sequence_bar = r"""
  <section class="yca__sequence-bar" aria-label="连续帧区间">
    <label>连续组<select id="yca-group-select"></select></label>
    <label>起点帧<input id="yca-start-frame" type="number" step="1"></label>
    <label>结束帧<input id="yca-end-frame" type="number" step="1"></label>
    <button class="yca__button yca__button--primary" id="yca-confirm-range" type="button">确认区间</button>
    <output id="yca-group-status">选择连续组后确认标注范围</output>
  </section>
"""
    html = html.replace("保存标注</button>", "保存当前</button><button class=\"yca__button yca__sync-save\" id=\"yca-sync-save\" type=\"button\">同步后续帧</button>", 1)
    return html.replace("  <main class=\"yca__workspace\">", sequence_bar + "  <main class=\"yca__workspace\">")


def _component_css() -> str:
    return base.DATASET_ANNOTATION_CSS.replace("yda", "yca") + r"""
.yca__sequence-bar { align-items:end; background:#eef3ef; border-bottom:1px solid var(--line); display:grid; gap:8px; grid-template-columns:minmax(180px,1fr) 110px 110px auto minmax(220px,1.4fr); padding:9px 12px; }
.yca__sequence-bar output { align-self:center; color:var(--muted); font-size:11px; overflow-wrap:anywhere; }
.yca__record-actions { grid-template-columns:1fr 1fr 1fr; }
.yca__sync-save { background:#eef3ef!important; border-color:#8eaa99!important; color:#245943!important; }
@media (max-width:820px) { .yca__sequence-bar { grid-template-columns:1fr 1fr; } .yca__sequence-bar label:first-child,.yca__sequence-bar output { grid-column:1/-1; } }
"""


def _component_js() -> str:
    script = base.DATASET_ANNOTATION_JS.replace("yda", "yca")
    script = script.replace("ui_list_annotation_datasets", "ui_list_consecutive_annotation_datasets")
    script = script.replace("ui_open_annotation_dataset", "ui_open_consecutive_annotation_dataset")
    script = script.replace("ui_load_annotation_sample", "ui_load_consecutive_annotation_sample")
    script = script.replace("ui_save_annotation_sample", "ui_save_consecutive_annotation_sample")
    script = script.replace(
        "ui_set_annotation_sample_reviewed",
        "ui_set_consecutive_annotation_sample_reviewed",
    )
    script = script.replace(
        "const state = {dataset:null,samples:[]",
        "const state = {dataset:null,allSamples:[],currentGroup:null,samples:[]",
    )
    old_open = 'async function openDataset(){ const path=$("#yca-dataset-path").value.trim(); if(!path)return toast("请输入或选择数据集路径");if(state.dirty&&!window.confirm("当前修改尚未保存，确定打开其他数据集吗？"))return; $("#yca-status").textContent="正在扫描数据集..."; try{ const result=await server.ui_open_consecutive_annotation_dataset({dataset_path:path}); $("#yca-dataset-path").value=result.dataset_path;syncDatasetChoice(result.dataset_path);refreshDatasetOptions(result.dataset_path).catch(error=>toast(`刷新数据集失败：${error?.message||error}`));state.dataset=result;state.samples=result.samples;state.sample=null;state.dirty=false;renderSamples(); $("#yca-status").textContent=`已加载 ${result.sample_count} 条数据${result.error_count?`，${result.error_count} 条无法读取`:\'\'}`; await selectSample(result.samples[0].key); }catch(error){$("#yca-status").textContent="数据集打开失败";toast(error?.message||error);} }'
    new_open = r'''function activateGroup(groupId,selectFirst=true){const group=state.dataset?.groups.find(item=>item.group_id===groupId);if(!group)return;state.currentGroup=groupId;state.samples=state.allSamples.filter(item=>item.group_id===groupId);$("#yca-group-select").value=groupId;$("#yca-start-frame").value=group.selected_start_frame;$("#yca-end-frame").value=group.selected_end_frame;$("#yca-start-frame").min=group.selected_start_frame;$("#yca-start-frame").max=group.selected_end_frame;$("#yca-end-frame").min=group.selected_start_frame;$("#yca-end-frame").max=group.selected_end_frame;$("#yca-group-status").textContent=`原始 f${group.original_start_frame}-f${group.original_end_frame} · 当前 ${group.frame_count} 帧${group.propagated?" · 已从起点初始化":" · 保存起点帧后自动初始化后续帧"}`;state.sample=null;renderSamples();if(selectFirst&&state.samples.length)selectSample(state.samples[0].key);}
function installGroups(result,preferredGroup=""){state.dataset=result;state.allSamples=result.samples;const select=$("#yca-group-select");select.replaceChildren();result.groups.forEach(group=>select.add(new Option(`${group.name} · f${group.selected_start_frame}-f${group.selected_end_frame}`,group.group_id)));activateGroup(result.groups.some(group=>group.group_id===preferredGroup)?preferredGroup:result.groups[0].group_id,false);}
async function openDataset(){ const path=$("#yca-dataset-path").value.trim(); if(!path)return toast("请输入或选择数据集路径");if(state.dirty&&!window.confirm("当前修改尚未保存，确定打开其他数据集吗？"))return; $("#yca-status").textContent="正在扫描连续帧数据集..."; try{ const result=await server.ui_open_consecutive_annotation_dataset({dataset_path:path}); $("#yca-dataset-path").value=result.dataset_path;syncDatasetChoice(result.dataset_path);refreshDatasetOptions(result.dataset_path).catch(error=>toast(`刷新数据集失败：${error?.message||error}`));installGroups(result);state.dirty=false;$("#yca-status").textContent=`已加载 ${result.groups.length} 个连续组、${result.sample_count} 帧`;await selectSample(state.samples[0].key); }catch(error){$("#yca-status").textContent="连续帧数据集打开失败";toast(error?.message||error);} }'''
    if old_open not in script:
        raise RuntimeError("base annotation openDataset template changed")
    script = script.replace(old_open, new_open)

    old_save_start = 'try{const result=await server.ui_save_consecutive_annotation_sample({dataset_path:state.dataset.dataset_path,sample_key:state.sample.key,edit});'
    new_save_start = 'try{const savedKey=state.sample.key;const result=await server.ui_save_consecutive_annotation_sample({dataset_path:state.dataset.dataset_path,sample_key:savedKey,edit,propagate_remaining:false});'
    script = script.replace(old_save_start, new_save_start)
    script = script.replace(
        'toast("当前标注已保存");}catch(error)',
        'if(result.propagated_count){const refreshed=await server.ui_open_consecutive_annotation_dataset({dataset_path:state.dataset.dataset_path});installGroups(refreshed,state.currentGroup);await selectSample(savedKey);toast(`已保存，并初始化后续 ${result.propagated_count} 帧`);}else toast("当前标注已保存");}catch(error)',
        1,
    )
    bindings = r'''
$("#yca-sync-save").onclick=async()=>{if(!state.sample)return;if(state.activeLine!==null)finishLine();const edit={yoyo_visibility:$("#yca-yoyo-visibility").value,yoyo_not_visible_reason:$("#yca-yoyo-not-visible-reason").value,trick_orientation:$("#yca-trick-orientation").value,yoyo_bbox_pixel:state.bbox,string_visibility:$("#yca-string-visibility").value,string_polylines_pixel:state.lines,string_review_status:$("#yca-review-status").value,bbox_review_status:"reviewed",reviewer:$("#yca-reviewer").value,notes:$("#yca-notes").value};const remaining=state.samples.length-state.current-1;if(!window.confirm(`保存当前帧，并用其标注覆盖后续 ${remaining} 帧吗？`))return;$("#yca-validation").textContent="正在同步保存...";$("#yca-sync-save").disabled=true;try{const savedKey=state.sample.key;const result=await server.ui_save_consecutive_annotation_sample({dataset_path:state.dataset.dataset_path,sample_key:savedKey,edit,propagate_remaining:true});state.dirty=false;const refreshed=await server.ui_open_consecutive_annotation_dataset({dataset_path:state.dataset.dataset_path});installGroups(refreshed,state.currentGroup);await selectSample(savedKey);$("#yca-validation").textContent="";toast(`当前帧已保存，并同步 ${result.propagated_count} 个后续帧`);}catch(error){$("#yca-validation").textContent=error?.message||String(error);}finally{$("#yca-sync-save").disabled=false;}};
$("#yca-group-select").addEventListener("change",event=>{if(state.dirty&&!window.confirm("当前修改尚未保存，确定切换连续组吗？")){event.target.value=state.currentGroup;return;}activateGroup(event.target.value);});
$("#yca-confirm-range").onclick=async()=>{if(!state.dataset||!state.currentGroup)return;if(state.dirty)return toast("请先保存当前修改");const start=Number($("#yca-start-frame").value),end=Number($("#yca-end-frame").value);if(!Number.isInteger(start)||!Number.isInteger(end))return toast("起点帧和结束帧必须是整数");if(!window.confirm(`确认将当前连续组保留为 f${start} 到 f${end} 吗？范围外帧会从连续组元数据中移除。`))return;$("#yca-confirm-range").disabled=true;try{const refreshed=await server.ui_select_consecutive_group_range({dataset_path:state.dataset.dataset_path,group_id:state.currentGroup,start_frame:start,end_frame:end});installGroups(refreshed,state.currentGroup);await selectSample(state.samples[0].key);toast(`已保留 ${state.samples.length} 帧`);}catch(error){toast(`区间保存失败：${error?.message||error}`);}finally{$("#yca-confirm-range").disabled=false;}};
'''
    return script.replace(
        '$("#yca-zoom").addEventListener("input",event=>setZoom(Number(event.target.value)));',
        bindings + '$("#yca-zoom").addEventListener("input",event=>setZoom(Number(event.target.value)));',
    )


def consecutive_annotation_component_kwargs() -> dict[str, Any]:
    return {
        "value": _component_html(),
        "css_template": _component_css(),
        "js_on_load": _component_js(),
        "apply_default_css": False,
        "container": False,
        "padding": False,
        "server_functions": [
            ui_list_consecutive_annotation_datasets,
            ui_open_consecutive_annotation_dataset,
            ui_load_consecutive_annotation_sample,
            ui_save_consecutive_annotation_sample,
            ui_set_consecutive_annotation_sample_reviewed,
            ui_select_consecutive_group_range,
        ],
    }
