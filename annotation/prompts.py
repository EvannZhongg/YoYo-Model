YOYO_DETECTION_PROMPT = """Detect every yoyo in the image and return only a valid JSON array.

Output format:
[
  {
    "bbox_2d": [x1, y1, x2, y2],
    "label": "yoyo",
    "sub_label": "visible yoyo body"
  }
]

Rules:
- bbox_2d must use normalized coordinates on a 0-999 scale: x1, y1, x2, y2.
- The box should tightly cover the visible yoyo body, including both halves if visible.
- Do not include hands, string, sticks, background objects, or motion blur unless they are part of the visible yoyo body.
- If no yoyo is visible, return [].
- Return JSON only. Do not include Markdown, comments, explanations, or extra text.
"""


VIDEO_FRAME_ANNOTATION_PROMPT = """Annotate one frame from a yoyo performance. Return exactly one valid JSON object.

Output schema:
{
  "visibility": "visible | partially_visible | occluded | out_of_frame | absent | uncertain",
  "yoyo_bbox_2d": [x1, y1, x2, y2] or null,
  "string_visibility": "visible | partial | not_visible | uncertain",
  "string_polylines_2d": [[[x, y], ...], ...] or null,
  "hands_2d": {"left": [x, y] or null, "right": [x, y] or null},
  "bad_case": ["code", ...],
  "notes": "short factual note"
}

Rules:
- All coordinates use the 0-999 normalized image coordinate system.
- yoyo_bbox_2d tightly covers only the visible yoyo body. Use null if no body is visible.
- string_polylines_2d contains every distinct visible string stroke. Use separate strokes for occlusion gaps, loops, crossings, or formations that cannot be represented by one unbranched polyline.
- Each stroke follows the visible string centerline and has at least two points. Add points at bends/intersections. Never invent hidden string geometry.
- If only part of the string is visible, label only that part and set string_visibility to partial.
- Do not infer occluded versus out_of_frame when evidence is insufficient; use uncertain.
- Suggested bad_case codes: yoyo_not_visible, yoyo_edge_clipped, motion_blur, string_not_visible, string_ambiguous, hands_occluded, multiple_yoyo, non_trick_scene.
- Intro, exit, audience, title card, and setup frames should use visibility absent and bad_case non_trick_scene.
- Return JSON only, without Markdown or explanations.
"""


EXAMPLE_PROMPTS = [
    YOYO_DETECTION_PROMPT,
    """Detect every car in this image and identify each one's color, report results in JSON format like this:
[
  {"bbox_2d": [x1, y1, x2, y2], "label": "car", "sub_label": "the car's color"}
]""",
    "detect all people in the image",
    "",
]
