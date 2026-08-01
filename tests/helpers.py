import json
from pathlib import Path

from PIL import Image

from workbench.consecutive_annotation import CONSECUTIVE_FILENAME, CONSECUTIVE_SCHEMA_VERSION
from workbench.dataset_annotation import ANNOTATION_SCHEMA_VERSION


def make_annotation_dataset(root: Path, name: str = "review_set") -> tuple[Path, str]:
    dataset = root / name
    group = "performer-01"
    image_path = dataset / "canonical" / "images" / group / "frame-001.jpg"
    label_path = dataset / "canonical" / "labels" / group / "frame-001.json"
    image_path.parent.mkdir(parents=True)
    label_path.parent.mkdir(parents=True)
    Image.new("RGB", (400, 200), "white").save(image_path)
    label_path.write_text(
        json.dumps(
            {
                "schema_version": ANNOTATION_SCHEMA_VERSION,
                "source_image": "../../images/performer-01/frame-001.jpg",
                "image_size": [400, 200],
                "source_group": group,
                "frame_index": 12,
                "visibility": "visible",
                "trick_orientation": "normal",
                "yoyo_bbox_pixel": [100, 50, 140, 90],
                "string_visibility": "partial",
                "string_polylines_pixel": [[[10, 20], [30, 40]]],
                "string_review_status": "unresolved",
                "string_path": {"topology": "single_path", "paths": []},
            }
        ),
        encoding="utf-8",
    )
    return dataset, f"{group}/frame-001.json"


def make_consecutive_dataset(root: Path, frame_count: int = 4) -> tuple[Path, list[str]]:
    dataset = root / "sequence-set"
    group = "video-a"
    keys = []
    frames = []
    for offset in range(frame_count):
        frame_index = 10 + offset
        stem = f"frame-{frame_index:03d}"
        image = dataset / "canonical" / "images" / group / f"{stem}.jpg"
        label = dataset / "canonical" / "labels" / group / f"{stem}.json"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (200, 100), (240 - offset, 240, 240)).save(image)
        key = f"{group}/{stem}.json"
        keys.append(key)
        label.write_text(
            json.dumps(
                {
                    "schema_version": ANNOTATION_SCHEMA_VERSION,
                    "source_image": f"../../images/{group}/{stem}.jpg",
                    "image_size": [200, 100],
                    "source_group": group,
                    "frame_index": frame_index,
                    "visibility": "visible",
                    "trick_orientation": "normal",
                    "yoyo_bbox_pixel": [10, 10, 30, 30],
                    "string_visibility": "partial",
                    "string_polylines_pixel": [[[5, 5], [25, 25]]],
                    "string_review_status": "unresolved",
                    "string_path": {"topology": "single_path", "paths": []},
                }
            ),
            encoding="utf-8",
        )
        frames.append(
            {
                "sample_key": key,
                "image": f"canonical/images/{group}/{stem}.jpg",
                "frame_index": frame_index,
                "timestamp_s": offset / 30,
            }
        )
    (dataset / CONSECUTIVE_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": CONSECUTIVE_SCHEMA_VERSION,
                "dataset_id": dataset.name,
                "groups": [
                    {
                        "group_id": "video-a--run-10-13",
                        "source_group": group,
                        "source_video": "video-a.mp4",
                        "original_start_frame": 10,
                        "original_end_frame": 13,
                        "selected_start_frame": 10,
                        "selected_end_frame": 13,
                        "start_sample_key": keys[0],
                        "frames": frames,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return dataset, keys
