from __future__ import annotations


BARADAD_SMALL_SCALE_14 = [
    "dead_leaves-squares",
    "dead_leaves-oriented",
    "dead_leaves-mixed",
    "dead_leaves-textures",
    "stat-spectrum",
    "stat-wmm",
    "stat-spectrum_color",
    "stat-spectrum_color_wmm",
    "stylegan-random",
    "stylegan-highfreq",
    "stylegan-sparse",
    "stylegan-oriented",
    "feature_vis-random",
    "feature_vis-dead_leaves",
]


DEFAULT_SCORE_ALIASES = {
    "dead_leaves-squares": "dead_squares",
    "dead_leaves-oriented": "dead_oriented",
    "dead_leaves-mixed": "dead_mixed",
    "dead_leaves-textures": "dead_textures",
    "stat-spectrum": "stat_spectrum",
    "stat-wmm": "stat_wmm",
    "stat-spectrum_color": "stat_spectrum_color",
    "stat-spectrum_color_wmm": "stat_spectrum_color_wmm",
    "stylegan-random": "stylegan_random",
    "stylegan-highfreq": "stylegan_highfreq",
    "stylegan-sparse": "stylegan_sparse",
    "stylegan-oriented": "stylegan_oriented",
    "feature_vis-random": "feature_vis_random",
    "feature_vis-dead_leaves": "feature_vis_dead_leaves",
}


MANUSCRIPT_SMALL_SCALE_SCORE_DATASETS = ["imagenet", *DEFAULT_SCORE_ALIASES.values()]


def guth_train_experiment_name(dataset: str, color: bool) -> str:
    safe_name = dataset.replace("-", "_")
    run_kind = "color" if color else "gray"
    return f"guth_small_{safe_name}_{run_kind}_64"
