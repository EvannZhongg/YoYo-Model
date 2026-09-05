# Temporal merge contract

The input batch is a temporal extraction dataset carrying
`temporal_review.json` (`yoyo_temporal_review_v1`). A source group is eligible
only when its entry has `status: confirmed` and at least three
`selected_sample_keys`. Every selected key must occur in the matching group in
`consecutive_groups.json` and resolve to a canonical label/image pair.

The aggregate `datasets/1Ayoyo_temporal` keeps:

- canonical images and `agent_yoyo_string_annotation_v5` labels;
- one manifest record per selected frame;
- source video SHA-256 and frame index provenance;
- groups containing the selected frames in source order, including gaps;
- generation-run metadata identifying each imported source manifest.

It does not keep `temporal_review.json`, `status`, reviewer identity, or group
confirmation timestamps. Those fields authorize import from an individual
annotation batch but are not part of the aggregate training data contract.

An aggregate group may have non-consecutive source frame indices. It must
contain at least three frames in strictly increasing order. Train/validation/
test partitioning must treat a complete group as one atomic unit.
