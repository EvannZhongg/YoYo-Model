"""Temporal-group annotation workflow for skill-generated datasets.

This module deliberately reuses the canonical and consecutive annotation
components, but exposes only datasets produced by the temporal extraction
skill (or the future aggregate ``1Ayoyo_temporal`` dataset).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common.files import atomic_write_text
from workbench import consecutive_annotation as consecutive
from workbench import dataset_annotation as base


TEMPORAL_SAMPLING_METHOD = "evenly_spaced_non_overlapping_consecutive_groups"
TEMPORAL_MARKER = "within_group_repeated_frames_allowed"
TEMPORAL_REVIEW_FILENAME = "temporal_review.json"
TEMPORAL_REVIEW_SCHEMA = "yoyo_temporal_review_v1"
MIN_SELECTED_FRAMES = 3


def _is_temporal_dataset(dataset: Path) -> bool:
    if dataset.name == "1Ayoyo_temporal":
        return True
    manifest_path = dataset / "manifest.json"
    sampling_path = dataset / "sampling_manifest.json"
    if not manifest_path.is_file() or not sampling_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sampling = json.loads(sampling_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        manifest.get("schema_version") == "yoyo_consecutive_annotation_dataset_v1"
        and manifest.get("deduplication", {}).get(TEMPORAL_MARKER) is True
        and sampling.get("sampling_method") == TEMPORAL_SAMPLING_METHOD
    )


def _read_temporal_metadata(dataset_path: str | Path) -> tuple[Path, dict[str, Any]]:
    dataset = base._managed_dataset_path(dataset_path)
    if not _is_temporal_dataset(dataset):
        raise ValueError(
            "Temporal 标注页只接受 skill 生成的 temporal 数据集或 datasets/1Ayoyo_temporal"
        )
    return consecutive._read_groups(dataset, allow_gaps=True)


def _review_path(dataset: Path) -> Path:
    return dataset / TEMPORAL_REVIEW_FILENAME


def _read_temporal_review(dataset: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    path = _review_path(dataset)
    if not path.is_file():
        document = {
            "schema_version": TEMPORAL_REVIEW_SCHEMA,
            "dataset_id": dataset.name,
            "created_at_utc": base._utc_now(),
            "updated_at_utc": base._utc_now(),
            "minimum_selected_frames": MIN_SELECTED_FRAMES,
            "groups": {},
        }
    else:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid temporal review metadata: {path}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != TEMPORAL_REVIEW_SCHEMA:
        raise ValueError(f"unsupported temporal review schema: {path}")
    if document.get("dataset_id") != dataset.name or not isinstance(document.get("groups"), dict):
        raise ValueError(f"temporal review metadata does not match dataset: {path}")
    group_ids = {str(group["group_id"]) for group in metadata["groups"]}
    for group_id, review in document["groups"].items():
        if group_id not in group_ids or not isinstance(review, dict):
            raise ValueError(f"temporal review references an unknown group: {group_id}")
    for group in metadata["groups"]:
        group_id = str(group["group_id"])
        frames = group["frames"]
        keys = [str(frame["sample_key"]) for frame in frames]
        entry = document["groups"].setdefault(group_id, {
            "status": "unresolved",
            "selected_sample_keys": keys,
            "selected_frame_indices": [int(frame["frame_index"]) for frame in frames],
            "reviewer": None,
            "confirmed_at_utc": None,
        })
        selected = entry.get("selected_sample_keys")
        if not isinstance(selected, list) or not set(str(value) for value in selected).issubset(keys):
            raise ValueError(f"temporal review selection is invalid: {group_id}")
        if entry.get("status") not in {"unresolved", "confirmed"}:
            raise ValueError(f"temporal review status is invalid: {group_id}")
        expected_indices = [
            int(frame["frame_index"]) for frame in frames
            if str(frame["sample_key"]) in {str(value) for value in selected}
        ]
        if entry.get("selected_frame_indices") is not None and entry.get("selected_frame_indices") != expected_indices:
            raise ValueError(f"temporal review frame indices disagree with selection: {group_id}")
        if entry.get("status") == "confirmed" and len(selected) < MIN_SELECTED_FRAMES:
            raise ValueError(f"confirmed temporal group has fewer than {MIN_SELECTED_FRAMES} frames: {group_id}")
    return document


def list_temporal_annotation_datasets() -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for item in base.list_annotation_datasets(include_consecutive=True):
        path = Path(item["path"])
        if not _is_temporal_dataset(path):
            continue
        try:
            _read_temporal_metadata(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        results.append(item)
    return results


def _temporal_group_summary(
    group: dict[str, Any], samples: list[dict[str, Any]], selected_keys: set[str]
) -> dict[str, Any]:
    summary = consecutive._group_summary(group)
    group_id = summary["group_id"]
    scoped = [
        sample for sample in samples
        if sample.get("group_id") == group_id
    ]
    reviewed_count = sum(bool(sample.get("reviewed")) for sample in scoped)
    summary.update(
        {
            "reviewed_count": reviewed_count,
            "unreviewed_count": len(scoped) - reviewed_count,
            "review_progress": round(reviewed_count / max(1, len(scoped)), 4),
            "frame_count": len(scoped),
        }
    )
    return summary


def open_temporal_annotation_dataset(dataset_path: str) -> dict[str, Any]:
    dataset, metadata = _read_temporal_metadata(dataset_path)
    review = _read_temporal_review(dataset, metadata)
    opened = consecutive.open_consecutive_annotation_dataset(str(dataset), allow_gaps=True)
    selected_by_group = {
        group_id: set(str(value) for value in entry.get("selected_sample_keys") or [])
        for group_id, entry in review["groups"].items()
    }
    opened["samples"] = [
        {
            **sample,
            "temporal_selected": str(sample.get("key"))
            in selected_by_group.get(str(sample.get("group_id")), set()),
        }
        for sample in opened["samples"]
    ]
    groups = []
    for group in metadata["groups"]:
        selected_keys = selected_by_group[str(group["group_id"])]
        summary = _temporal_group_summary(group, opened["samples"], selected_keys)
        entry = review["groups"][str(group["group_id"])]
        summary.update(
            {
                "group_review_status": str(entry.get("status") or "unresolved"),
                "selected_sample_keys": list(entry.get("selected_sample_keys") or []),
                "selected_frame_indices": list(entry.get("selected_frame_indices") or [
                    int(frame["frame_index"])
                    for frame in group["frames"]
                    if str(frame["sample_key"]) in selected_keys
                ]),
                "selected_frame_count": len(entry.get("selected_sample_keys") or []),
                "minimum_selected_frames": MIN_SELECTED_FRAMES,
            }
        )
        groups.append(summary)
    return {
        **opened,
        "dataset_type": "temporal",
        "groups": groups,
        "temporal_group_count": len(groups),
        "temporal_reviewed_group_count": sum(item["group_review_status"] == "confirmed" for item in groups),
        "temporal_review_path": str(_review_path(dataset)),
        "metadata_path": str(dataset / consecutive.CONSECUTIVE_FILENAME),
    }


def save_temporal_annotation_sample(
    dataset_path: str,
    sample_key: str,
    edit_json: str | dict[str, Any],
    propagate_remaining: bool = False,
) -> dict[str, Any]:
    dataset, metadata = _read_temporal_metadata(dataset_path)
    if not any(
        str(sample_key) == str(frame.get("sample_key"))
        for group in metadata["groups"]
        for frame in group["frames"]
    ):
        raise ValueError("sample is outside the temporal group mapping")
    return consecutive.save_consecutive_annotation_sample(
        dataset_path, sample_key, edit_json, propagate_remaining
    )


def set_temporal_group_reviewed(
    dataset_path: str,
    group_id: str,
    reviewer: str,
    confirmed: bool = True,
) -> dict[str, Any]:
    dataset, metadata = _read_temporal_metadata(dataset_path)
    review = _read_temporal_review(dataset, metadata)
    group = next(
        (item for item in metadata["groups"] if str(item.get("group_id")) == str(group_id)),
        None,
    )
    if group is None:
        raise ValueError("temporal group was not found")
    reviewer_name = str(reviewer or "workbench-reviewer")
    entry = review["groups"][str(group_id)]
    selected_keys = [str(value) for value in entry.get("selected_sample_keys") or []]
    if confirmed:
        _, labels_root, _ = base._annotation_roots(dataset)
        reviews = base._dataset_reviews(base._read_review_map(), dataset)
        reviewed_keys = {
            str(frame["sample_key"])
            for frame in group["frames"]
            if base._review_summary(
                labels_root / str(frame["sample_key"]),
                reviews.get(str(frame["sample_key"])),
            )["reviewed"]
        }
        selected_keys = [
            str(frame["sample_key"])
            for frame in group["frames"]
            if str(frame["sample_key"]) in reviewed_keys
        ]
        # Compatibility for datasets created by older clients that explicitly
        # persisted a selection before frame-review metadata was introduced.
        if len(selected_keys) < MIN_SELECTED_FRAMES and not reviews and len(entry.get("selected_sample_keys") or []) >= MIN_SELECTED_FRAMES:
            selected_keys = [str(value) for value in entry["selected_sample_keys"]]
        if len(selected_keys) < MIN_SELECTED_FRAMES:
            raise ValueError(f"本组至少需要 {MIN_SELECTED_FRAMES} 张人工核验完成的帧后才能确认")
        entry["selected_sample_keys"] = selected_keys
        entry["selected_frame_indices"] = [
            int(frame["frame_index"])
            for frame in group["frames"]
            if str(frame["sample_key"]) in set(selected_keys)
        ]
    # Group confirmation is intentionally independent from single-frame review
    # bindings.  Per-frame confirmations are written only through the frame
    # review action and are carried by the merge skill when present.
    updated = 0
    entry["status"] = "confirmed" if confirmed else "unresolved"
    entry["reviewer"] = reviewer_name if confirmed else None
    entry["confirmed_at_utc"] = base._utc_now() if confirmed else None
    review["updated_at_utc"] = base._utc_now()
    atomic_write_text(
        _review_path(dataset),
        json.dumps(review, ensure_ascii=False, indent=2) + "\n",
    )
    refreshed = open_temporal_annotation_dataset(str(dataset))
    return {
        **refreshed,
        "group_id": str(group_id),
        "confirmed": confirmed,
        "updated_count": updated,
    }


def ui_list_temporal_annotation_datasets(_payload: object = None) -> list[dict[str, str]]:
    return list_temporal_annotation_datasets()


def ui_open_temporal_annotation_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    return open_temporal_annotation_dataset(str((payload or {}).get("dataset_path") or ""))


def ui_load_temporal_annotation_sample(payload: dict[str, Any]) -> dict[str, Any]:
    _read_temporal_metadata(str((payload or {}).get("dataset_path") or ""))
    return base.load_annotation_sample(
        str((payload or {}).get("dataset_path") or ""),
        str((payload or {}).get("sample_key") or ""),
    )


def ui_save_temporal_annotation_sample(payload: dict[str, Any]) -> dict[str, Any]:
    return save_temporal_annotation_sample(
        str((payload or {}).get("dataset_path") or ""),
        str((payload or {}).get("sample_key") or ""),
        (payload or {}).get("edit") or {},
        bool((payload or {}).get("propagate_remaining", False)),
    )


def ui_set_temporal_annotation_sample_reviewed(payload: dict[str, Any]) -> dict[str, Any]:
    _read_temporal_metadata(str((payload or {}).get("dataset_path") or ""))
    return base.set_annotation_sample_reviewed(
        str((payload or {}).get("dataset_path") or ""),
        str((payload or {}).get("sample_key") or ""),
        str((payload or {}).get("reviewer") or ""),
        bool((payload or {}).get("confirmed", True)),
    )


def ui_set_temporal_group_reviewed(payload: dict[str, Any]) -> dict[str, Any]:
    return set_temporal_group_reviewed(
        str((payload or {}).get("dataset_path") or ""),
        str((payload or {}).get("group_id") or ""),
        str((payload or {}).get("reviewer") or ""),
        bool((payload or {}).get("confirmed", True)),
    )


def set_temporal_group_selection(
    dataset_path: str,
    group_id: str,
    selected_sample_keys: list[str],
) -> dict[str, Any]:
    dataset, metadata = _read_temporal_metadata(dataset_path)
    review = _read_temporal_review(dataset, metadata)
    group = next((item for item in metadata["groups"] if str(item["group_id"]) == str(group_id)), None)
    if group is None:
        raise ValueError("temporal group was not found")
    available = {str(frame["sample_key"]) for frame in group["frames"]}
    selected = list(dict.fromkeys(str(key) for key in selected_sample_keys))
    if not set(selected).issubset(available):
        raise ValueError("temporal selection contains a frame outside the group")
    if len(selected) < MIN_SELECTED_FRAMES:
        raise ValueError(f"每组至少保留 {MIN_SELECTED_FRAMES} 帧")
    entry = review["groups"][str(group_id)]
    entry["selected_sample_keys"] = selected
    entry["selected_frame_indices"] = [
        int(frame["frame_index"]) for frame in group["frames"] if str(frame["sample_key"]) in set(selected)
    ]
    if entry.get("status") == "confirmed":
        entry["status"] = "unresolved"
        entry["reviewer"] = None
        entry["confirmed_at_utc"] = None
    review["updated_at_utc"] = base._utc_now()
    atomic_write_text(_review_path(dataset), json.dumps(review, ensure_ascii=False, indent=2) + "\n")
    return open_temporal_annotation_dataset(str(dataset))


def ui_set_temporal_group_selection(payload: dict[str, Any]) -> dict[str, Any]:
    return set_temporal_group_selection(
        str((payload or {}).get("dataset_path") or ""),
        str((payload or {}).get("group_id") or ""),
        list((payload or {}).get("selected_sample_keys") or []),
    )


def ui_select_temporal_group_range(payload: dict[str, Any]) -> dict[str, Any]:
    """Compatibility endpoint kept for the shared editor template.

    Temporal clips are fixed by the extractor, so the UI hides this control;
    the endpoint still validates the dataset if an old browser calls it.
    """
    _read_temporal_metadata(str((payload or {}).get("dataset_path") or ""))
    return consecutive.select_consecutive_group_range(
        str((payload or {}).get("dataset_path") or ""),
        str((payload or {}).get("group_id") or ""),
        int((payload or {}).get("start_frame")),
        int((payload or {}).get("end_frame")),
    )


def _component_html() -> str:
    html = consecutive._component_html().replace("yca", "yta")
    html = html.replace("连续帧标注", "Temporal 组级标注", 1)
    dashboard = r'''
      <section class="yta__group-dashboard" aria-label="Temporal 组管理">
        <output id="yta-group-progress">组进度</output>
        <div class="yta__group-actions">
          <button class="yta__button" id="yta-prev-group" type="button">上一组</button>
          <button class="yta__button" id="yta-next-group" type="button">下一组</button>
          <button class="yta__button yta__button--primary" id="yta-review-group" type="button">确认本组可用</button>
        </div>
      </section>
'''
    return html.replace('      <ol class="yta__sample-list" id="yta-sample-list"></ol>\n    </aside>', '      <ol class="yta__sample-list" id="yta-sample-list"></ol>\n' + dashboard + '    </aside>')


def _component_css() -> str:
    css = consecutive._component_css().replace("yca", "yta")
    return css + r'''
.yta__group-dashboard { background:#f4f8f4; border-top:1px solid var(--line); display:grid; gap:7px; padding:9px 10px 10px; }
.yta__sidebar { grid-template-rows:auto auto minmax(0,1fr) auto; }
.yta__group-dashboard output { color:var(--muted); font-size:11px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.yta__group-actions { display:grid; gap:6px; grid-template-columns:1fr 1fr; }
.yta__group-actions .yta__button--primary { grid-column:1/-1; }
.yta__group-actions .yta__button { min-height:32px; padding:6px 7px; }
.yta__sample-list button.is-discarded { opacity:.58; }
.yta__sample-list button.is-discarded::after { color:var(--muted); content:'未保留'; float:right; font-size:10px; }
@media (max-width:820px) { .yta__group-actions { grid-template-columns:1fr 1fr; } }
'''


def _component_js() -> str:
    script = consecutive._component_js().replace("yca", "yta")
    replacements = {
        "ui_list_consecutive_annotation_datasets": "ui_list_temporal_annotation_datasets",
        "ui_open_consecutive_annotation_dataset": "ui_open_temporal_annotation_dataset",
        "ui_load_consecutive_annotation_sample": "ui_load_temporal_annotation_sample",
        "ui_save_consecutive_annotation_sample": "ui_save_temporal_annotation_sample",
        "ui_set_consecutive_annotation_sample_reviewed": "ui_set_temporal_annotation_sample_reviewed",
        "ui_select_consecutive_group_range": "ui_select_temporal_group_range",
    }
    for old, new in replacements.items():
        script = script.replace(old, new)
    # Temporal groups are fixed clips; range trimming is intentionally not
    # exposed. Keep the existing group selector and add group-level progress.
    script = script.replace(
        '$("#yta-confirm-range").onclick=async()=>{',
        '$("#yta-confirm-range").hidden=true;$("#yta-start-frame").readOnly=true;$("#yta-end-frame").readOnly=true;$("#yta-confirm-range").onclick=async()=>{',
        1,
    )
    dashboard = r'''
function syncTemporalGroupDashboard(){const groups=state.dataset?.groups||[],index=groups.findIndex(item=>item.group_id===state.currentGroup),group=index>=0?groups[index]:null;if(!group)return;const selected=group.selected_frame_count,total=group.frame_count,progress=group.group_review_status==="confirmed"?"组已确认":`${group.reviewed_count}/${total} 帧单独核验`;$("#yta-group-progress").textContent=`第 ${index+1}/${groups.length} 组 · 已选 ${selected}/${total} 帧 · ${progress}`;$("#yta-prev-group").disabled=index<=0;$("#yta-next-group").disabled=index<0||index>=groups.length-1;const button=$("#yta-review-group"),done=group.group_review_status==="confirmed";button.textContent=done?"取消组确认":"确认本组可用";button.classList.toggle("is-reviewed",done);button.disabled=!group||state.dirty;}
function activateTemporalGroup(index){const groups=state.dataset?.groups||[];if(index<0||index>=groups.length)return;activateGroup(groups[index].group_id);syncTemporalGroupDashboard();}
'''
    script = script.replace("function activateGroup(groupId,selectFirst=true){", dashboard + "function activateGroup(groupId,selectFirst=true){", 1)
    script = script.replace(
        'result.groups.forEach(group=>select.add(new Option(`${group.name} · f${group.selected_start_frame}-f${group.selected_end_frame}`,group.group_id)))',
        'result.groups.forEach(group=>select.add(new Option(`${group.name} · ${group.reviewed_count||0}/${group.frame_count||0}${group.group_review_status==="confirmed"?"， ✓":""} · f${group.selected_start_frame}-f${group.selected_end_frame}`,group.group_id)))',
        1,
    )
    script = script.replace(
        'result.groups.forEach(group=>select.add(new Option(`${group.name} · f${group.selected_start_frame}-f${group.selected_end_frame}`,group.group_id)));',
        'result.groups.forEach(group=>select.add(new Option(`${group.name} · ${group.reviewed_count||0}/${group.frame_count||0}${group.group_review_status==="confirmed"?"， ✓":""} · f${group.selected_start_frame}-f${group.selected_end_frame}`,group.group_id)));',
        1,
    )
    script = script.replace(
        "state.currentGroup=groupId;state.samples=state.allSamples.filter(item=>item.group_id===groupId);",
        "state.currentGroup=groupId;const activeGroup=state.dataset?.groups.find(item=>item.group_id===groupId);state.selectedKeys=new Set(activeGroup?.selected_sample_keys||[]);state.samples=state.allSamples.filter(item=>item.group_id===groupId);",
        1,
    )
    script = script.replace(
        'class="${state.sample?.key===sample.key?\'is-active\':\'\'}"',
        'class="${state.sample?.key===sample.key?\'is-active\':\'\'}"',
        1,
    )
    script = script.replace("state.sample=null;renderSamples();if(selectFirst&&state.samples.length)selectSample(state.samples[0].key);}", "state.sample=null;renderSamples();syncTemporalGroupDashboard();if(selectFirst&&state.samples.length)selectSample(state.samples[0].key);}", 1)
    script = script.replace("activateGroup(result.groups.some(group=>group.group_id===preferredGroup)?preferredGroup:result.groups[0].group_id,false);}", "activateGroup(result.groups.some(group=>group.group_id===preferredGroup)?preferredGroup:result.groups[0].group_id,false);syncTemporalGroupDashboard();}", 1)
    script = script.replace(
        'state.samples[index]={...state.samples[index],...result};$("#yta-dirty")',
        'state.samples[index]={...state.samples[index],...result};state.allSamples=state.allSamples.map(item=>item.key===result.key?{...item,...result}:item);const activeGroup=state.dataset.groups.find(item=>item.group_id===state.currentGroup);if(activeGroup){activeGroup.reviewed_count=Math.max(0,activeGroup.reviewed_count+(confirmed?1:-1));activeGroup.unreviewed_count=Math.max(0,activeGroup.frame_count-activeGroup.reviewed_count);}temporalSampleCache.delete(result.key);syncTemporalGroupDashboard();$("#yta-dirty")',
        1,
    )
    script = script.replace(
        'const canvas = $("#yta-canvas");',
        'const canvas = $("#yta-canvas");const temporalImageCache=new Map();const temporalSampleCache=new Map();const temporalPrefetch=path=>{if(!path||temporalImageCache.has(path))return;const image=new Image();image.src=fileUrl(path);temporalImageCache.set(path,image);};const temporalSamplePrefetch=key=>{if(!key||temporalSampleCache.has(key)||!state.dataset)return;const pending=server.ui_load_temporal_annotation_sample({dataset_path:state.dataset.dataset_path,sample_key:key});temporalSampleCache.set(key,pending);pending.catch(()=>temporalSampleCache.delete(key));};',
        1,
    )
    script = script.replace(
        'const image=new Image(); image.onload=()=>{if(request!==state.loadSerial)return;state.image=image;canvas.classList.add("is-ready");$("#yta-empty").hidden=true;renderCanvas();}; image.onerror=()=>{if(request===state.loadSerial){$("#yta-empty").textContent="图像加载失败";toast("图像加载失败");}}; image.src=fileUrl(result.image_path);',
        'const image=temporalImageCache.get(result.image_path)||new Image();temporalImageCache.set(result.image_path,image);const showImage=()=>{if(request!==state.loadSerial)return;state.image=image;canvas.classList.add("is-ready");$("#yta-empty").hidden=true;renderCanvas();}; image.onload=showImage; image.onerror=()=>{if(request===state.loadSerial){$("#yta-empty").textContent="图像加载失败";toast("图像加载失败");}}; if(image.complete&&image.naturalWidth)showImage();else image.src=fileUrl(result.image_path); const sampleIndex=state.samples.findIndex(item=>item.key===key);[state.samples[sampleIndex-1],state.samples[sampleIndex+1]].forEach(item=>{temporalPrefetch(item?.image_path);temporalSamplePrefetch(item?.key);});',
        1,
    )
    script = script.replace(
        'state.dirty=false;const refreshed=await server.ui_open_temporal_annotation_dataset({dataset_path:state.dataset.dataset_path});installGroups(refreshed,state.currentGroup);await selectSample(savedKey);',
        'state.dirty=false;temporalSampleCache.clear();const refreshed=await server.ui_open_temporal_annotation_dataset({dataset_path:state.dataset.dataset_path});installGroups(refreshed,state.currentGroup);await selectSample(savedKey);',
        1,
    )
    script = script.replace(
        'const result=await server.ui_load_temporal_annotation_sample({dataset_path:state.dataset.dataset_path,sample_key:key});',
        'let result=temporalSampleCache.get(key);if(!result){result=server.ui_load_temporal_annotation_sample({dataset_path:state.dataset.dataset_path,sample_key:key});temporalSampleCache.set(key,result);}result=await result;',
        1,
    )
    script = script.replace(
        'state.sample.annotation=result.annotation;state.sample.reviewed=false;',
        'temporalSampleCache.delete(savedKey);state.sample.annotation=result.annotation;state.sample.reviewed=false;',
        1,
    )
    bindings = r'''
$("#yta-prev-group").onclick=()=>{const index=(state.dataset?.groups||[]).findIndex(item=>item.group_id===state.currentGroup);activateTemporalGroup(index-1);};
$("#yta-next-group").onclick=()=>{const index=(state.dataset?.groups||[]).findIndex(item=>item.group_id===state.currentGroup);activateTemporalGroup(index+1);};
$("#yta-review-group").onclick=async()=>{if(!state.dataset||!state.currentGroup||state.dirty)return;const group=state.dataset.groups.find(item=>item.group_id===state.currentGroup),confirmed=group.group_review_status!=="confirmed";if(confirmed&&group.reviewed_count<3)return toast("本组至少需要 3 张人工核验完成的帧");if(!window.confirm(`${confirmed?"确认":"取消"}本组可用吗？`))return;const button=$("#yta-review-group");button.disabled=true;try{const currentKey=state.sample?.key;const result=await server.ui_set_temporal_group_reviewed({dataset_path:state.dataset.dataset_path,group_id:state.currentGroup,reviewer:$("#yta-reviewer").value,confirmed});installGroups(result,state.currentGroup);syncTemporalGroupDashboard();if(currentKey)selectSample(currentKey);const updated=result.groups.find(item=>item.group_id===state.currentGroup);toast(confirmed?`本组已确认，自动选中 ${updated?.selected_frame_count||0} 张已核验帧`:"已取消本组确认");}catch(error){toast(`组级审核失败：${error?.message||error}`);}finally{button.disabled=false;syncTemporalGroupDashboard();}};
'''
    return script.replace(
        '$("#yta-zoom").addEventListener("input",event=>setZoom(Number(event.target.value)));',
        bindings + '$("#yta-zoom").addEventListener("input",event=>setZoom(Number(event.target.value)));',
        1,
    )


def temporal_annotation_component_kwargs() -> dict[str, Any]:
    from workbench.preannotation import ui_preannotate_dataset

    return {
        "value": _component_html(),
        "css_template": _component_css(),
        "js_on_load": _component_js(),
        "apply_default_css": False,
        "container": False,
        "padding": False,
        "server_functions": [
            ui_list_temporal_annotation_datasets,
            ui_open_temporal_annotation_dataset,
            ui_load_temporal_annotation_sample,
            ui_save_temporal_annotation_sample,
            ui_set_temporal_annotation_sample_reviewed,
            ui_set_temporal_group_reviewed,
            ui_set_temporal_group_selection,
            ui_select_temporal_group_range,
            ui_preannotate_dataset,
        ],
    }
