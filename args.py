import argparse


def parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="data/", help="raw data path")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )

    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument(
        "--wandb_pj_name", type=str, default="EDGE", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )

    # ---- style / attribute conditioning ----
    parser.add_argument(
        "--num_styles",
        type=int,
        default=0,
        help="Number of discrete dance styles (0 disables style conditioning)",
    )
    parser.add_argument(
        "--attr_list",
        type=str,
        default="",
        help="Comma-separated continuous dance attributes, e.g. 'energy,sleeve_amp,com_sway'",
    )
    parser.add_argument(
        "--style_drop_prob", type=float, default=0.2, help="style dropout for CFG"
    )
    parser.add_argument(
        "--attr_drop_prob", type=float, default=0.2, help="attribute dropout for CFG"
    )

    # ---- costume physics proxy losses (module 3) ----
    parser.add_argument(
        "--wrist_smooth_weight",
        type=float,
        default=0.0,
        help="Weight of wrist-trajectory smoothness loss",
    )
    parser.add_argument(
        "--arc_weight",
        type=float,
        default=0.0,
        help="Weight of wrist arc (direction-change) loss, proxy for circular sleeve motion",
    )
    parser.add_argument(
        "--penetration_weight",
        type=float,
        default=0.0,
        help="Weight of arm-torso penetration proxy loss",
    )
    parser.add_argument(
        "--penetration_thresh",
        type=float,
        default=0.05,
        help="Distance threshold (normalized space) for penetration penalty",
    )
    opt = parser.parse_args()
    return opt


def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument("--out_length", type=float, default=30, help="max. length of output, in seconds")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint.pt", help="checkpoint"
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="data/test/wavs",
        help="folder containing input music",
    )
    parser.add_argument(
        "--save_motions", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        default="eval/motions",
        help="Where to save the motions",
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Save the jukebox features for later reuse",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Don't render the video",
    )
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="cached_features/",
        help="Where to save/load the features",
    )

    # ---- style / attribute conditioning at inference ----
    parser.add_argument(
        "--num_styles",
        type=int,
        default=0,
        help="Number of discrete dance styles; overrides the checkpoint value if the checkpoint has none",
    )
    parser.add_argument(
        "--style",
        type=int,
        default=-1,
        help="Style id to generate with (-1 = unconditional)",
    )
    parser.add_argument(
        "--attrs",
        type=str,
        default="",
        help="Comma-separated continuous attribute values, e.g. '1.2,0.8' (order follows --attr_list at training time)",
    )
    parser.add_argument(
        "--style_guidance",
        type=float,
        default=1.0,
        help="CFG weight for the style condition",
    )
    parser.add_argument(
        "--attr_guidance",
        type=float,
        default=1.0,
        help="CFG weight for the attribute condition",
    )

    # ---- long-form generation ----
    parser.add_argument(
        "--mode",
        type=str,
        choices=["long", "sequential"],
        default="long",
        help="long: batched windows with hard overlap copying; sequential: autoregressive windows conditioned on the previous window tail",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=30,
        help="Frames (30 fps) of the previous window used to condition the next one in sequential mode",
    )
    parser.add_argument(
        "--structure_aware",
        action="store_true",
        help="Derive per-window guidance weights from music onset density",
    )
    opt = parser.parse_args()
    return opt
