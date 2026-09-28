# WaterSleeve &mdash; Style-Controllable Dance Generation for Chinese Intangible Cultural Heritage

**WaterSleeve** generates dance from music with discrete *style* control (Dunhuang / Chinese classical / ethnic folk dance), continuous *attribute* control (energy, sleeve amplitude, COM sway), costume-aware physics regularization, and long-form (full-song) generation. It is built on [EDGE](https://arxiv.org/abs/2211.10658) (Editable Dance Generation From Music, CVPR 2023) and fully preserves EDGE's native editing capabilities (joint-wise conditioning, in-betweening).

> 中国传统非遗舞蹈的风格化可控生成：音乐驱动，风格/属性双重条件，服饰物理约束，完整作品级长序列生成，并完整保留 EDGE 原生可编辑能力。

## What's new over EDGE

| Module | Description | Code |
|---|---|---|
| **Style & attribute conditioning** | Discrete style embeddings + continuous dance attributes (劲道 energy, 水袖幅度 sleeve amplitude, 重心起伏 COM sway) with independent classifier-free guidance per condition | [model/model.py](model/model.py) |
| **Self-supervised attribute labels** | Attributes are derived from FK positions automatically when the dataset has no annotations | [model/diffusion.py](model/diffusion.py) `compute_motion_attrs` |
| **Costume physics proxy losses** | Wrist-trajectory smoothness, circular (water-sleeve) arc motion, arm-torso penetration &mdash; differentiable kinematic constraints in the diffusion loss, extending the PFC heuristic idea | [model/diffusion.py](model/diffusion.py) `p_losses` |
| **Long-form generation** | Sequential autoregressive windows with cross-window consistency (inpainting constraints), plus onset-density-aware guidance per music section | [model/diffusion.py](model/diffusion.py) `sequential_sample`, [data/audio_extraction/structure.py](data/audio_extraction/structure.py) |
| **Style transfer & dance repair** | Reuses EDGE's native inpainting / joint conditioning machinery | [dataset/masks.py](dataset/masks.py) |
| **Evaluation suite** | Kinematic style features + style classifier, beat-alignment metric, extended physical-plausibility metrics (cloth proxies) | [eval/](eval/) |

The full competition plan is in [竞赛方案.md](竞赛方案.md). New flags are fully backward compatible: without them, training and inference behave exactly like the original EDGE.

## Requirements

* We recommend Linux for performance and compatibility reasons. Windows will probably work, but is not officially supported.
* 64-bit Python 3.7+
* PyTorch 1.12.1 (newer versions, e.g. 2.x, also work)
* At least 16 GB RAM per GPU
* 1&ndash;8 high-end NVIDIA GPUs with at least 16 GB of GPU memory, NVIDIA drivers, CUDA 11.6 toolkit.
* Optional: `librosa` (beat-alignment metric, music-structure guidance), `scipy` (motion onset detection)

This repository additionally depends on the following libraries, which may require special installation procedures:
* [jukemirlib](https://github.com/rodrigo-castellon/jukemirlib)
* [pytorch3d](https://github.com/facebookresearch/pytorch3d)
* [accelerate](https://huggingface.co/docs/accelerate/v0.16.0/en/index)
	* Note: after installation, don't forget to run `accelerate config` . We use fp16.
* [wine](https://www.winehq.org) (Optional, for import to Blender only)

## Getting started

* Download the saved model checkpoint from [Google Drive](https://drive.google.com/file/d/1BAR712cVEqB8GR37fcEihRV_xOC-fZrZ/view?usp=share_link) or by running `bash download_model.sh`.
* Run `demo.ipynb`, which demonstrates the basic interface of the model.

### Load custom music

You can test the model on custom music by downloading them as `.wav` files into a directory, e.g. `custom_music/` and running

```.bash
python test.py --music_dir custom_music/
```

This process may take a while, since the script will extract all the Jukebox representations for the specified music in memory. The representations can also be saved and reused to improve speed with the `--cache_features` and `--use_cached_features` arguments. See `args.py` for more detail.
Note: make sure file names are regularized, e.g. `Britney Spears - Toxic (Official HD Video).wav` may cause unpredictable behavior due to the spaces and parentheses, but `toxic.wav` will behave as expected. See how the demo notebook achieves this using the `youtube-dl --output` flag.

## Style & attribute conditioned generation

With a checkpoint that has style/attribute heads (see training below):

```.bash
# style 1, attribute values (energy, sleeve_amp, com_sway) = (1.2, 0.8, 0.5)
python test.py --music_dir custom_music/ --checkpoint checkpoint.pt \
    --style 1 --attrs 1.2,0.8,0.5 \
    --style_guidance 1.0 --attr_guidance 1.0
```

Omit `--style` (or pass a negative id) for unconditional generation; omit `--attrs` to drop attribute conditioning.

## Long-form generation

```.bash
# sequential mode: each 5s window is conditioned on the previous window's tail
python test.py --music_dir custom_music/ --mode sequential --overlap 75 \
    --structure_aware --out_length 60
```

* `--mode long` (default): EDGE's original batched windows with hard overlap copying.
* `--mode sequential`: autoregressive windows with inpainting-based cross-window consistency (new).
* `--overlap`: number of frames (30 fps) from the previous window used to condition the next. `75` matches the 2.5 s slice stride used by `test.py`.
* `--structure_aware`: derive per-window guidance weights from music onset density (requires `librosa`).

## Training

Once the AIST++ dataset is downloaded and processed (see below), run the training script, e.g.

```.bash
accelerate launch train.py --batch_size 128  --epochs 2000 --feature_type jukebox --learning_rate 0.0002
```

### Fine-tuning with style / attribute heads and costume physics losses

Start from any EDGE checkpoint (the new heads are initialized randomly; everything else loads as-is):

```.bash
accelerate launch train.py --batch_size 128 --epochs 2000 \
    --checkpoint checkpoint.pt \
    --num_styles 3 \
    --attr_list energy,sleeve_amp,com_sway \
    --wrist_smooth_weight 0.1 --arc_weight 0.1 \
    --penetration_weight 0.05 --penetration_thresh 0.05
```

* `--num_styles 0` (default) disables the style head; `--attr_list ""` disables the attribute head.
* Attribute labels are computed from ground-truth motion automatically (self-supervised) until a labeled dataset is available.
* `num_styles` / `attr_list` are stored in the checkpoint and restored automatically at inference time.
* The training will log progress to `wandb` and intermittently produce sample outputs to visualize learning.

### (Optional, retraining only) Dataset Download

Download and process the AIST++ dataset (wavs and motion only) using:

```.bash
cd data
bash download_dataset.sh
python create_dataset.py --extract-baseline --extract-jukebox
```

This will process the dataset to match the settings used in the paper. The data processing will take ~24 hrs and ~50 GB to precompute all the Jukebox features for the dataset.

## Evaluation

Evaluate your model's outputs with the Physical Foot Contact (PFC) score proposed in the paper:

```.bash
python test.py --music_dir custom_music/ --save_motions
python eval/eval_pfc.py
```

New evaluation axes (module 6):

```.bash
# physical plausibility incl. costume physics proxies (wrist smoothness / arc / penetration)
python eval/eval_pfc.py --motion_path eval/motions --mode all

# style fidelity: kinematic style features, intra/inter-style separation, classifier
python eval/eval_style.py --motion_path eval/motions_by_style/
python eval/eval_style.py --motion_path eval/motions_by_style/ --train_classifier style_clf.pt

# musicality: beat alignment (requires librosa, motion dir + matching wav dir)
python eval/eval_beat.py --motion_dir eval/motions --wav_dir custom_music/
```

## Smoke tests

The new conditioning, proxy-loss and sequential-sampling plumbing can be verified without pytorch3d/vis (they are replaced by minimal stubs):

```.bash
python tests/smoke_test.py
```

## Blender 3D rendering

In order to render generated dances in 3D, we convert them into FBX files to be used in Blender. We provide a sample rig, `SMPL-to-FBX/ybot.fbx`.
After generating dances with the `--save-motions` flag enabled, move the relevant saved `.pkl` files to a folder, e.g. `smpl_samples`

```.bash
python SMPL-to-FBX/Convert.py --input_dir SMPL-to-FBX/smpl_samples/ --output_dir SMPL-to-FBX/fbx_out
```

to convert motions into FBX files, which can be imported into Blender and retargeted onto different rigs, i.e. from [Mixamo](https://www.mixamo.com). A variety of retargeting tools are available, such as the [Rokoko plugin for Blender](https://www.rokoko.com/integrations/blender).

## Citation

If you use WaterSleeve, please cite the original EDGE paper:

```
@article{tseng2022edge,
  title={EDGE: Editable Dance Generation From Music},
  author={Tseng, Jonathan and Castellon, Rodrigo and Liu, C. Karen},
  journal={arXiv preprint arXiv:2211.10658},
  year={2022}
}
```

## Acknowledgements

WaterSleeve builds on the official EDGE implementation by the Stanford Movement Lab. We would like to thank [lucidrains](https://github.com/lucidrains) for the [Adan](https://github.com/lucidrains/Adan-pytorch) and [diffusion](https://github.com/lucidrains/denoising-diffusion-pytorch) repos, [softcat477](https://github.com/softcat477) for their [SMPL to FBX](https://github.com/softcat477/SMPL-to-FBX) library, and [BobbyAnguelov](https://github.com/BobbyAnguelov) for their [FBX Converter tool](https://github.com/BobbyAnguelov/FbxFormatConverter).
