# VGGT-Omega FiftyOne Zoo Model

<p align="center">
  <img src="vggt-omega.gif" alt="VGGT-Omega demo"/>
</p>


[VGGT-Omega](https://vggt-omega.github.io/) (CVPR 2026) by Meta AI and Oxford VGG takes a video and reconstructs the 3D scene from it — estimating depth for every frame and building a single merged 3D point cloud, all in one forward pass. No iterative refinement, no Structure-from-Motion pipeline.

## What you get after running the model

For each video in your dataset:

| Output | Where it lives | What it is |
|---|---|---|
| Per-frame depth map | `sample.frames[i]["depth_map"]` | `fo.Heatmap` rendered as an overlay in the App |
| Merged 3D scene | `sample["scene_3d"]` | Path to a `.fo3d` file viewable in FiftyOne's 3D viewer |

## Prerequisites

Install the model package and dependencies before registering the zoo source:

```bash
pip install git+https://github.com/facebookresearch/vggt-omega.git
pip install fiftyone open3d einops safetensors huggingface_hub opencv-python
```

## Step 1 — Register the zoo source

```python
import fiftyone.zoo as foz

foz.register_zoo_model_source(
    "https://github.com/harpreetsahota204/vggt_omega",
    overwrite=True,
)
```

## Step 2 — Load the model

```python
model = foz.load_zoo_model(
    "facebook/VGGT-Omega-1B-512",
    confidence_threshold=50.0,      # percentile for point cloud filtering (0–100)
    video_sample_fps=1.0,           # target frames/sec to sample from each video
    max_frames=50,                  # hard cap on frames per forward pass
    preprocessing_mode="balanced",  # "balanced" or "max_size"
    image_resolution=512,           # do not change for this checkpoint
)
```

**Parameters at a glance:**

| Parameter | Default | Notes |
|---|---|---|
| `confidence_threshold` | `50.0` | Percentile (0–100) for filtering low-confidence depth points from the merged point cloud. Higher → fewer but more reliable points |
| `video_sample_fps` | `1.0` | Target frames per second to sample. Auto-reduced for long videos to stay within `max_frames`. Set to your video's native fps to get depth on every frame |
| `max_frames` | `50` | Hard cap on frames per forward pass. Based on official benchmarks (A100, 624×416 inputs): ~7GB for 16 frames, ~10GB for 50, ~13GB for 100, ~21GB for 200 |
| `preprocessing_mode` | `"balanced"` | `"balanced"` keeps total token count ≈ `image_resolution²`. `"max_size"` resizes the longest side to `image_resolution` — lower memory per frame |
| `image_resolution` | `512` | Tokeniser resolution — fixed for the 1B-512 checkpoint, do not change. Use `256` only with the 256-Text checkpoint |

## Step 3 — Compute metadata and run inference

`compute_metadata()` must be called first so the model can read each video's frame rate and duration.

```python
import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart-video", persistent=True)

dataset.compute_metadata()
dataset.apply_model(model, "depth_map")
```

After this call:
- `sample.frames[i]["depth_map"]` — depth Heatmap for each sampled frame
- `sample["scene_3d"]` — path to the `.fo3d` 3D scene file

## Step 4 — Build a grouped dataset for viewing

To explore depth maps and the 3D scene side-by-side in the FiftyOne App, build a grouped dataset:

```python
def build_vggt_omega_grouped_dataset(source_dataset, name="vggt_omega_results", overwrite=True):
    from pathlib import Path

    grouped = fo.Dataset(name, overwrite=overwrite)
    grouped.add_group_field("group", default="video")
    samples = []

    for source_sample in source_dataset.iter_samples(progress=True):
        path = Path(source_sample.filepath)
        group = fo.Group()

        video_sample = fo.Sample(filepath=str(path), group=group.element("video"))
        for i, depth_png in enumerate(sorted(path.parent.glob(f"{path.stem}_frame_*_depth.png"))):
            video_sample.frames[i + 1]["depth_map"] = fo.Heatmap(map_path=str(depth_png), range=[0, 255])
        samples.append(video_sample)

        fo3d = path.parent / f"{path.stem}_scene.fo3d"
        if fo3d.exists():
            samples.append(fo.Sample(filepath=str(fo3d), group=group.element("threed")))

    grouped.add_samples(samples)
    return grouped


grouped_dataset = build_vggt_omega_grouped_dataset(dataset)
```

## Step 5 — Explore in the App

```python
session = fo.launch_app(grouped_dataset)
```

- Switch between the **video** slice to see depth heatmap overlays on each frame
- Switch to the **threed** slice to open the merged 3D point cloud in the 3D viewer

## Text-alignment checkpoint

A second checkpoint produces a scene-level embedding alongside depth and poses, suitable for scene-similarity search.

```python
model = foz.load_zoo_model(
    "facebook/VGGT-Omega-1B-256-Text",
    image_resolution=256,     # required for this checkpoint
    enable_alignment=True,
    confidence_threshold=50.0,
    video_sample_fps=2.0,
    max_frames=16,
)

dataset.compute_metadata()
dataset.apply_model(model, "depth_map")

# Each sample now also has a 2048-D scene embedding
print(dataset.first()["text_alignment_embedding"])
```

Use `compute_similarity` to index the embeddings for nearest-neighbour scene search:

```python
import fiftyone.brain as fob

fob.compute_similarity(
    dataset,
    embeddings="text_alignment_embedding",
    brain_key="scene_sim",
)
```

## Citation

```bibtex
@inproceedings{wang2026vggtomega,
  title={VGGT-{$\Omega$}},
  author={Wang, Jianyuan and Chen, Minghao and Zhang, Shangzhan and Karaev, Nikita and
          Sch{\"o}nberger, Johannes and Labatut, Patrick and Bojanowski, Piotr and
          Novotny, David and Vedaldi, Andrea and Rupprecht, Christian},
  booktitle={CVPR},
  year={2026}
}
```
