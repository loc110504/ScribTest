import torch


@torch.no_grad()
def mean_scribble_confidence(prob_map, scribble_label, ignore_index=4):
    if prob_map.ndim != 4:
        raise ValueError(f"prob_map must have shape [B, C, H, W], got {tuple(prob_map.shape)}")
    if scribble_label.ndim != 3:
        raise ValueError(
            f"scribble_label must have shape [B, H, W], got {tuple(scribble_label.shape)}"
        )
    if prob_map.shape[0] != scribble_label.shape[0] or prob_map.shape[2:] != scribble_label.shape[1:]:
        raise ValueError(
            "prob_map and scribble_label must share the same batch/spatial shape, got "
            f"{tuple(prob_map.shape)} and {tuple(scribble_label.shape)}"
        )

    valid_mask = scribble_label.ne(ignore_index)
    valid_count = valid_mask.sum()
    if valid_count.item() == 0:
        return {
            "mean_confidence": prob_map.new_zeros(()),
            "confidence_map": prob_map.new_zeros(scribble_label.shape),
            "valid_mask": valid_mask,
            "valid_count": valid_count,
        }

    safe_label = scribble_label.clone()
    safe_label[~valid_mask] = 0
    confidence_map = torch.gather(prob_map, dim=1, index=safe_label.unsqueeze(1)).squeeze(1)
    mean_confidence = confidence_map[valid_mask].mean()

    return {
        "mean_confidence": mean_confidence,
        "confidence_map": confidence_map,
        "valid_mask": valid_mask,
        "valid_count": valid_count,
    }


@torch.no_grad()
def select_teacher_by_scribble_confidence(
    teacher1_prob,
    teacher2_prob,
    scribble_label,
    ignore_index=4,
):
    teacher1_stats = mean_scribble_confidence(
        prob_map=teacher1_prob,
        scribble_label=scribble_label,
        ignore_index=ignore_index,
    )
    teacher2_stats = mean_scribble_confidence(
        prob_map=teacher2_prob,
        scribble_label=scribble_label,
        ignore_index=ignore_index,
    )

    teacher1_conf = teacher1_stats["mean_confidence"]
    teacher2_conf = teacher2_stats["mean_confidence"]
    teacher1_wins = teacher1_conf.item() >= teacher2_conf.item()

    return {
        "mode": 1 if teacher1_wins else 2,
        "winner_key": "teacher1" if teacher1_wins else "teacher2",
        "winner_confidence": teacher1_conf if teacher1_wins else teacher2_conf,
        "teacher1_confidence": teacher1_conf,
        "teacher2_confidence": teacher2_conf,
        "valid_count": teacher1_stats["valid_count"],
        "valid_mask": teacher1_stats["valid_mask"],
    }
