"""
VGGT-Omega FiftyOne Remote Source Zoo Model
============================================
Video-first wrapper for facebook/VGGT-Omega (CVPR 2026).

Processes all sampled frames of a video in a single VGGT-Omega forward pass,
producing:
  • Per-frame grayscale depth PNGs  → stored as fo.Heatmap in sample.frames[i][label_field]
  • Merged multi-frame 3D point cloud (.pcd + .fo3d) → stored in sample["scene_3d"]

Requires dataset.compute_metadata() to be run before dataset.apply_model().
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch

import fiftyone as fo
import fiftyone.core.models as fom
import fiftyone.utils.torch as fout

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import pose_encoding_to_extri_intri

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class VGGTOmegaModelConfig(fout.TorchImageModelConfig):
    """Configuration for the VGGT-Omega FiftyOne zoo model.

    Args:
        model_name (str): manifest ``base_name`` (e.g. ``"facebook/VGGT-Omega-1B-512"``).
        model_path (str): path to the downloaded .pt state-dict checkpoint.
        confidence_threshold (float): depth-confidence percentile (0–100) for
            point cloud filtering. Default 50.0.
        video_sample_fps (float): target frames per second to extract from the
            input video. Auto-reduced for long videos to stay within max_frames.
            Default 2.0.
        max_frames (int): hard cap on frames fed to VGGT-Omega per forward pass.
            Keeps VRAM bounded regardless of video length. Default 16.
        image_resolution (int): tokeniser resolution — 512 for the 1B-512
            checkpoint, 256 for the 1B-256-Text checkpoint. Default 512.
        enable_alignment (bool): activate the TextAlignmentHead and store a
            2048-D L2-normalised scene embedding in sample["text_alignment_embedding"].
            Only valid with the 256-Text checkpoint. Default False.
    """

    def __init__(self, d):
        super().__init__(d)
        # Prevents FiftyOne's image-loading pipeline from decoding the video.
        self.raw_inputs = True

        self.model_name = self.parse_string(d, "model_name", default="facebook/VGGT-Omega-1B-512")
        self.model_path = self.parse_string(d, "model_path")
        self.confidence_threshold = self.parse_number(d, "confidence_threshold", default=50.0)
        self.video_sample_fps = self.parse_number(d, "video_sample_fps", default=2.0)
        self.max_frames = self.parse_number(d, "max_frames", default=16)
        self.image_resolution = self.parse_number(d, "image_resolution", default=512)
        self.enable_alignment = self.parse_bool(d, "enable_alignment", default=False)

        print(
            f"[VGGTOmegaModelConfig] Initialised:\n"
            f"  model_name        = {self.model_name}\n"
            f"  model_path        = {self.model_path}\n"
            f"  confidence_thresh = {self.confidence_threshold}\n"
            f"  video_sample_fps  = {self.video_sample_fps}\n"
            f"  max_frames        = {self.max_frames}\n"
            f"  image_resolution  = {self.image_resolution}\n"
            f"  enable_alignment  = {self.enable_alignment}"
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VGGTOmegaModel(fom.Model, fom.SamplesMixin):
    """FiftyOne zoo model wrapper for VGGT-Omega.

    Designed for video datasets. FiftyOne passes an ``FFmpegVideoReader`` to
    ``predict()`` per sample; the model reads only ``.inpath`` from it and
    uses cv2 for frame extraction.

    Outputs per sample:
      • ``sample.frames[i][label_field]`` — fo.Heatmap pointing to a
        per-frame grayscale depth PNG (1-based frame number, sampled frames only)
      • ``sample["scene_3d"]``             — path to the merged .fo3d scene
      • ``sample["text_alignment_embedding"]`` — 2048-D list (256-Text only)

    Requires ``dataset.compute_metadata()`` before ``dataset.apply_model()``.

    Args:
        config (VGGTOmegaModelConfig): model configuration.
    """

    def __init__(self, config: VGGTOmegaModelConfig):
        self.config = config
        self._fields: Dict = {}

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            cap = torch.cuda.get_device_capability()
            self._dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            self._device = torch.device("cpu")
            self._dtype = torch.float32

        print(f"[VGGTOmegaModel] Device: {self._device}  |  dtype: {self._dtype}")

        model = VGGTOmega(enable_alignment=config.enable_alignment)
        state_dict = torch.load(config.model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        self._model = model.to(self._device).eval()
        print(
            f"[VGGTOmegaModel] Loaded {sum(p.numel() for p in self._model.parameters()):,} params"
        )

    # ------------------------------------------------------------------
    # FiftyOne interface
    # ------------------------------------------------------------------

    @property
    def media_type(self) -> str:
        return "video"

    @property
    def ragged_batches(self) -> bool:
        return True

    @property
    def transforms(self):
        return None

    @property
    def preprocess(self) -> bool:
        return False

    @property
    def needs_fields(self) -> Dict:
        return self._fields

    @needs_fields.setter
    def needs_fields(self, fields: Dict):
        self._fields = fields

    # ------------------------------------------------------------------
    # Context manager — clears GPU cache after apply_model completes
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        return False

    # ------------------------------------------------------------------
    # FiftyOne entry point
    # ------------------------------------------------------------------

    def predict(self, arg, sample=None):
        """Called by FiftyOne for each video sample.

        Returns a dict of ``{1-based frame number: fo.Heatmap}`` entries.
        Sample-level fields (scene_3d, text_alignment_embedding) are set
        directly on the sample object so FiftyOne's ctx.save() persists them.
        """
        filepath = arg.inpath if hasattr(arg, "inpath") else sample.filepath
        print(f"[predict] filepath: {filepath}")
        return self._process_video(filepath, sample)

    # ------------------------------------------------------------------
    # Core video processing
    # ------------------------------------------------------------------

    def _process_video(self, filepath: str, sample) -> Dict:
        """Extract frames → VGGT-Omega forward pass → save outputs → return frame labels."""
        video_path = Path(filepath)
        output_dir = video_path.parent
        stem = video_path.stem

        print(f"\n{'='*60}\n[_process_video] {filepath}")

        if sample.metadata is None:
            raise RuntimeError(
                f"sample.metadata is None for '{filepath}'. "
                "Run dataset.compute_metadata() before applying this model."
            )
        meta = sample.metadata
        print(
            f"[_process_video] fps={meta.frame_rate:.2f}  "
            f"frames={meta.total_frame_count}  duration={meta.duration:.2f}s"
        )

        with tempfile.TemporaryDirectory(prefix="vggt_omega_") as tmpdir:
            frame_paths = self._extract_frames(
                filepath, tmpdir,
                video_fps=meta.frame_rate,
                duration_s=meta.duration,
                total_frame_count=meta.total_frame_count,
            )
            print(f"[_process_video] Extracted {len(frame_paths)} frames")

            images = load_and_preprocess_images(
                frame_paths,
                image_resolution=int(self.config.image_resolution),
            ).to(self._device)
            print(f"[_process_video] Preprocessed shape: {tuple(images.shape)}")

            with torch.inference_mode():
                predictions = self._model(images)
            print(f"[_process_video] Prediction keys: {list(predictions.keys())}")

            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                predictions["pose_enc"],
                predictions["images"].shape[-2:],
            )

            depth_all  = predictions["depth"].detach().float().cpu().numpy()      # [1, N, H, W, 1]
            conf_all   = predictions["depth_conf"].detach().float().cpu().numpy() # [1, N, H, W]
            extri_all  = extrinsics.detach().float().cpu().numpy()                # [1, N, 3, 4]
            intri_all  = intrinsics.detach().float().cpu().numpy()                # [1, N, 3, 3]
            images_np  = predictions["images"].detach().float().cpu().numpy()     # [1, N, 3, H, W]
            n_frames   = depth_all.shape[1]

            label_dict: Dict = {}
            all_points: List[np.ndarray] = []
            all_colors: List[np.ndarray] = []

            for i in range(n_frames):
                depth_i = depth_all[0, i, :, :, 0]
                conf_i  = conf_all[0, i]
                print(
                    f"[_process_video] Frame {i:03d}: "
                    f"depth [{depth_i.min():.3f}, {depth_i.max():.3f}]  "
                    f"conf [{conf_i.min():.3f}, {conf_i.max():.3f}]"
                )

                depth_png = output_dir / f"{stem}_frame_{i:06d}_depth.png"
                self._save_depth_png(depth_i, depth_png)
                label_dict[i + 1] = fo.Heatmap(map_path=str(depth_png), range=[0, 255])

                pts, cols = self._unproject_and_filter(
                    depth_i, conf_i, extri_all[0, i], intri_all[0, i], images_np[0, i],
                )
                if pts is not None:
                    all_points.append(pts)
                    all_colors.append(cols)
                    print(f"[_process_video] Frame {i:03d}: {pts.shape[0]:,} points kept")

            if all_points:
                scene_fo3d = output_dir / f"{stem}_scene.fo3d"
                self._save_scene(
                    np.concatenate(all_points), np.concatenate(all_colors), scene_fo3d
                )
                sample["scene_3d"] = str(scene_fo3d)
                print(f"[_process_video] scene_3d → {scene_fo3d}")
            else:
                logger.warning("[_process_video] No valid points — scene_3d not written.")

            if self.config.enable_alignment and "text_alignment_embedding" in predictions:
                emb = predictions["text_alignment_embedding"].detach().float().cpu().numpy()
                sample["text_alignment_embedding"] = emb.tolist()
                print(f"[_process_video] text_alignment_embedding shape: {emb.shape}")

            print(f"[_process_video] Done — {n_frames} depth maps saved.\n{'='*60}\n")
            return label_dict

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(
        self,
        video_path: str,
        output_dir: str,
        video_fps: float,
        duration_s: float,
        total_frame_count: int,
    ) -> List[str]:
        """Extract evenly-spaced frames from a video using cv2.

        Derives the effective sample rate from video_sample_fps and max_frames so
        that the total number of extracted frames never exceeds max_frames.

        Args:
            video_path: path to the video file
            output_dir: directory to write frame PNGs
            video_fps: native frame rate from VideoMetadata.frame_rate
            duration_s: duration in seconds from VideoMetadata.duration
            total_frame_count: total frame count from VideoMetadata.total_frame_count

        Returns:
            Sorted list of saved PNG file paths.
        """
        max_frames = int(self.config.max_frames)
        effective_fps = min(
            float(self.config.video_sample_fps),
            max_frames / max(duration_s, 1e-6),
        )
        effective_fps = max(effective_fps, 0.1)
        frame_interval = max(int(round(video_fps / effective_fps)), 1)

        print(
            f"[_extract_frames] effective_fps={effective_fps:.2f}  "
            f"frame_interval={frame_interval}  "
            f"(max_frames={max_frames}, total={total_frame_count})"
        )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frame_paths: List[str] = []
        frame_idx = 0
        saved_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_interval == 0:
                out_path = str(Path(output_dir) / f"{saved_idx:06d}.png")
                cv2.imwrite(out_path, frame)
                frame_paths.append(out_path)
                saved_idx += 1
            frame_idx += 1

        cap.release()
        print(f"[_extract_frames] Saved {len(frame_paths)} frames to {output_dir}")
        return frame_paths

    # ------------------------------------------------------------------
    # Depth map saving
    # ------------------------------------------------------------------

    def _save_depth_png(self, depth: np.ndarray, output_path: Path) -> None:
        """Normalise depth to [0, 255] and save as a single-channel grayscale PNG.

        Single-channel uint8 is required for fo.Heatmap(map_path=..., range=[0, 255])
        to render correctly in the FiftyOne App.

        Args:
            depth: raw depth values [H, W]
            output_path: destination PNG path
        """
        valid_mask = np.isfinite(depth) & (depth > 0)
        if not np.any(valid_mask):
            logger.warning(f"No valid depth values for {output_path}")
            cv2.imwrite(str(output_path), np.zeros_like(depth, dtype=np.uint8))
            return

        d_min = np.percentile(depth[valid_mask], 5)
        d_max = np.percentile(depth[valid_mask], 95)
        norm = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0.0, 1.0)
        norm[~valid_mask] = 0.0
        cv2.imwrite(str(output_path), (norm * 255).astype(np.uint8))

    # ------------------------------------------------------------------
    # Depth unprojection (ported from demo_gradio.py)
    # ------------------------------------------------------------------

    def _unproject_and_filter(
        self,
        depth: np.ndarray,      # [H, W]
        conf: np.ndarray,       # [H, W]
        extrinsic: np.ndarray,  # [3, 4]
        intrinsic: np.ndarray,  # [3, 3]
        color_chw: np.ndarray,  # [3, H, W]  values in [0, 1]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Unproject depth to world-space XYZ and filter by confidence percentile.

        Returns:
            (points [N, 3], colors [N, 3]) or (None, None) if no points survive.
        """
        H, W = depth.shape
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        fx, fy = intrinsic[0, 0], intrinsic[1, 1]
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]

        camera_pts = np.stack(
            [(x - cx) / fx * depth, (y - cy) / fy * depth, depth], axis=-1
        )  # [H, W, 3]

        R, T = extrinsic[:3, :3], extrinsic[:3, 3]
        world_pts = np.einsum("ij,hwj->hwi", R.T, camera_pts - T[None, None, :])

        threshold_val = np.percentile(conf, self.config.confidence_threshold)
        mask = (conf >= threshold_val) & np.isfinite(depth) & (depth > 0)

        if not np.any(mask):
            logger.warning("No points survived confidence filtering.")
            return None, None

        return (
            world_pts[mask].astype(np.float32),
            color_chw.transpose(1, 2, 0)[mask].astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Point cloud saving
    # ------------------------------------------------------------------

    def _save_scene(
        self,
        points: np.ndarray,  # [N, 3]
        colors: np.ndarray,  # [N, 3]  values in [0, 1]
        fo3d_path: Path,
    ) -> None:
        """Save the merged point cloud as a .pcd file and a FiftyOne .fo3d scene.

        Args:
            points: world-space XYZ [N, 3]
            colors: RGB colours in [0, 1] [N, 3]
            fo3d_path: destination .fo3d path (.pcd uses same stem)
        """
        pcd_path = fo3d_path.with_suffix(".pcd")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
        o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False)
        print(f"[_save_scene] PCD ({len(points):,} pts) → {pcd_path}")

        scene = fo.Scene()
        scene.camera = fo.PerspectiveCamera(up="Z")
        scene.add(fo.PointCloud("pointcloud", str(pcd_path)))
        scene.write(str(fo3d_path))
        print(f"[_save_scene] fo3d → {fo3d_path}")
