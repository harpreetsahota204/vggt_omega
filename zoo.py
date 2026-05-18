"""
VGGT-Omega FiftyOne Remote Source Zoo Model
============================================
Video-first wrapper for facebook/VGGT-Omega (CVPR 2026).

Processes all sampled frames of a video in a single VGGT-Omega forward pass,
producing:
  • Per-frame colorised depth PNGs  → stored in sample.frames[i]["depth_map_path"]
  • Merged multi-frame 3D point cloud (.pcd + .fo3d) → stored in label_field
"""

import logging
import os
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
        confidence_threshold (float): depth-confidence percentile (0–100) used
            to filter points when building the merged point cloud. Default 50.0.
        video_sample_fps (float): frames per second to extract from the input
            video. Default 2.0.
        image_resolution (int): tokeniser resolution passed to
            ``load_and_preprocess_images``. Use 512 for the 1B-512 checkpoint
            and 256 for the 1B-256-Text checkpoint. Default 512.
        enable_alignment (bool): activate the TextAlignmentHead and store a
            2048-D L2-normalised scene embedding. Only valid with the 256-Text
            checkpoint. Default False.
    """

    def __init__(self, d):
        super().__init__(d)
        # raw_inputs=True prevents FiftyOne's image-loading pipeline from
        # trying to decode the video file as a still image.
        self.raw_inputs = True

        self.model_name = self.parse_string(
            d, "model_name", default="facebook/VGGT-Omega-1B-512"
        )
        self.model_path = self.parse_string(d, "model_path")
        self.confidence_threshold = self.parse_number(
            d, "confidence_threshold", default=50.0
        )
        self.video_sample_fps = self.parse_number(
            d, "video_sample_fps", default=2.0
        )
        self.image_resolution = self.parse_number(
            d, "image_resolution", default=512
        )
        self.enable_alignment = self.parse_bool(
            d, "enable_alignment", default=False
        )

        print(
            f"[VGGTOmegaModelConfig] Initialised:\n"
            f"  model_name        = {self.model_name}\n"
            f"  model_path        = {self.model_path}\n"
            f"  confidence_thresh = {self.confidence_threshold}\n"
            f"  video_sample_fps  = {self.video_sample_fps}\n"
            f"  image_resolution  = {self.image_resolution}\n"
            f"  enable_alignment  = {self.enable_alignment}"
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class VGGTOmegaModel(fom.Model, fom.SamplesMixin):
    """FiftyOne zoo model wrapper for VGGT-Omega.

    Primary mode: video datasets.  For each video sample FiftyOne passes an
    ``FFmpegVideoReader`` to ``predict()``; the model ignores the reader and
    uses ``sample.filepath`` directly (cv2 frame extraction), matching the
    Gradio demo approach.

    Outputs stored by FiftyOne's ``apply_model``:
      • ``label_field``                     — path to merged .fo3d 3D scene
      • ``sample.frames[i]["depth_map_path"]`` — per-frame depth PNG paths

    Args:
        config (VGGTOmegaModelConfig): model configuration.
    """

    def __init__(self, config: VGGTOmegaModelConfig):
        self.config = config

        # Device & dtype selection
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            cap = torch.cuda.get_device_capability()
            self._dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            self._device = torch.device("cpu")
            self._dtype = torch.float32

        print(
            f"[VGGTOmegaModel] Device: {self._device}  |  dtype: {self._dtype}"
        )

        # Load model
        self._model = self._load_vggt_omega(config)

        # SamplesMixin requirement
        self._fields: Dict = {}

    # ------------------------------------------------------------------
    # FiftyOne required interface
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
    # Model loading
    # ------------------------------------------------------------------

    def _load_vggt_omega(self, config: VGGTOmegaModelConfig) -> VGGTOmega:
        """Load VGGTOmega from a state-dict checkpoint file."""
        if not os.path.exists(config.model_path):
            raise FileNotFoundError(
                f"[VGGTOmegaModel] Checkpoint not found: {config.model_path}"
            )

        print(
            f"[VGGTOmegaModel] Loading VGGTOmega "
            f"(enable_alignment={config.enable_alignment}) ..."
        )
        model = VGGTOmega(enable_alignment=config.enable_alignment)
        state_dict = torch.load(
            config.model_path, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict)
        model = model.to(self._device).eval()
        print(
            f"[VGGTOmegaModel] Model loaded on {self._device}. "
            f"Parameters: {sum(p.numel() for p in model.parameters()):,}"
        )
        return model

    # ------------------------------------------------------------------
    # FiftyOne entry point
    # ------------------------------------------------------------------

    def predict(self, arg, sample=None):
        """Entry point called by FiftyOne for each video sample.

        ``arg`` is an ``FFmpegVideoReader`` (with ``.inpath``) when FiftyOne
        routes through ``_apply_video_model``.  We extract only the filepath
        and use cv2 for frame extraction.

        Returns a mixed-key dict:
          - str keys  → stored at the sample level (e.g. merged scene fo3d)
          - int keys  → stored at frame level (1-based frame number)
        """
        # --- resolve filepath ---
        if hasattr(arg, "inpath"):
            filepath = arg.inpath
            print(f"[predict] Received FFmpegVideoReader, filepath: {filepath}")
        elif isinstance(arg, dict):
            filepath = arg.get("filepath") or (sample.filepath if sample else None)
            print(f"[predict] Received dict, filepath: {filepath}")
        elif isinstance(arg, str):
            filepath = arg
            print(f"[predict] Received str filepath: {filepath}")
        else:
            filepath = sample.filepath if sample else str(arg)
            print(f"[predict] Fallback filepath from sample: {filepath}")

        if filepath is None:
            logger.error("[predict] Could not determine video filepath; skipping.")
            return {}

        try:
            return self._process_video(filepath, sample)
        except Exception as exc:
            logger.error(
                f"[predict] Failed on '{filepath}': {exc}", exc_info=True
            )
            return {}

    # ------------------------------------------------------------------
    # Core video processing
    # ------------------------------------------------------------------

    def _process_video(self, filepath: str, sample) -> Dict:
        """Extract frames → run VGGT-Omega → save outputs → return label dict."""
        video_path = Path(filepath)
        output_dir = video_path.parent
        stem = video_path.stem

        print(f"\n{'='*60}")
        print(f"[_process_video] Processing: {filepath}")
        print(f"[_process_video] Output dir: {output_dir}")
        print(f"[_process_video] video_sample_fps: {self.config.video_sample_fps}")

        # Step 1 – extract frames to a temporary directory
        with tempfile.TemporaryDirectory(prefix="vggt_omega_") as tmpdir:
            frame_paths, frame_indices = self._extract_frames(
                filepath, tmpdir, self.config.video_sample_fps
            )

            if not frame_paths:
                logger.error(f"[_process_video] No frames extracted from {filepath}")
                return {}

            n_frames = len(frame_paths)
            print(f"[_process_video] Extracted {n_frames} frames → {tmpdir}")

            # Step 2 – preprocess with VGGT-Omega's loader
            print(
                f"[_process_video] Preprocessing frames at "
                f"image_resolution={self.config.image_resolution} ..."
            )
            images = load_and_preprocess_images(
                frame_paths,
                image_resolution=int(self.config.image_resolution),
            )
            print(f"[_process_video] Preprocessed tensor shape: {tuple(images.shape)}")
            images = images.to(self._device)

            # Step 3 – single VGGT-Omega forward pass
            print(f"[_process_video] Running VGGT-Omega forward pass ...")
            with torch.inference_mode():
                predictions = self._model(images)
            print(
                f"[_process_video] Forward pass complete. "
                f"Prediction keys: {list(predictions.keys())}"
            )

            # Step 4 – decode camera poses
            print(f"[_process_video] Decoding camera poses ...")
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                predictions["pose_enc"],
                predictions["images"].shape[-2:],  # (H, W) of preprocessed frames
            )
            print(
                f"[_process_video] extrinsics shape: {tuple(extrinsics.shape)}  "
                f"intrinsics shape: {tuple(intrinsics.shape)}"
            )

            # Move everything to CPU numpy for downstream processing
            depth_all = predictions["depth"].detach().float().cpu().numpy()     # [1, N, H, W, 1]
            conf_all = predictions["depth_conf"].detach().float().cpu().numpy() # [1, N, H, W]
            extri_all = extrinsics.detach().float().cpu().numpy()               # [1, N, 3, 4]
            intri_all = intrinsics.detach().float().cpu().numpy()               # [1, N, 3, 3]
            # images tensor for per-frame colour extraction
            images_np = predictions["images"].detach().float().cpu().numpy()    # [1, N, 3, H, W]

            print(
                f"[_process_video] depth_all shape: {depth_all.shape}  "
                f"conf_all shape: {conf_all.shape}"
            )

            # Step 5 – per-frame outputs
            label_dict: Dict = {}
            all_points: List[np.ndarray] = []
            all_colors: List[np.ndarray] = []

            for i in range(n_frames):
                depth_i = depth_all[0, i, :, :, 0]  # [H, W]
                conf_i  = conf_all[0, i]             # [H, W]
                extri_i = extri_all[0, i]            # [3, 4]
                intri_i = intri_all[0, i]            # [3, 3]
                color_i = images_np[0, i]            # [3, H, W]  values in [0,1]

                print(
                    f"[_process_video] Frame {i:03d}: "
                    f"depth range [{depth_i.min():.3f}, {depth_i.max():.3f}]  "
                    f"conf range [{conf_i.min():.3f}, {conf_i.max():.3f}]"
                )

                # Save colorised depth PNG
                depth_png_path = output_dir / f"{stem}_frame_{i:06d}_depth.png"
                self._save_depth_png(depth_i, depth_png_path)

                # Store Heatmap directly (not wrapped in a nested dict).
                # FiftyOne's add_labels sees non-dict values and uses label_field
                # as the frame field name, so frames end up with sample.frames[i][label_field].
                # Callers should use apply_model(model, "depth_map", ...) so the field
                # is named "depth_map" on every frame.
                fo_frame_num = i + 1
                label_dict[fo_frame_num] = fo.Heatmap(
                    map_path=str(depth_png_path),
                    range=[0, 255],
                )

                # Accumulate world-space points for merged cloud
                pts, cols = self._unproject_and_filter(
                    depth_i, conf_i, extri_i, intri_i, color_i,
                    self.config.confidence_threshold,
                )
                if pts is not None:
                    all_points.append(pts)
                    all_colors.append(cols)
                    print(
                        f"[_process_video] Frame {i:03d}: "
                        f"{pts.shape[0]:,} points after confidence filter"
                    )

            # Step 6 – build and save merged 3D scene
            scene_fo3d_path = output_dir / f"{stem}_scene.fo3d"
            if all_points:
                merged_pts = np.concatenate(all_points, axis=0)
                merged_cols = np.concatenate(all_colors, axis=0)
                print(
                    f"[_process_video] Merged point cloud: "
                    f"{merged_pts.shape[0]:,} total points"
                )
                self._save_merged_scene(
                    merged_pts, merged_cols, scene_fo3d_path
                )
            else:
                logger.warning(
                    "[_process_video] No valid points across all frames; "
                    "writing empty scene."
                )
                self._save_empty_scene(scene_fo3d_path)

            # Step 7 – store sample-level fields directly on the sample object.
            # FiftyOne's _apply_video_model calls ctx.save(sample) after predict()
            # returns, so any fields set here are persisted automatically.
            # We must NOT include sample-level string values in the returned dict
            # because add_labels() expects all values to be frame-level dicts when
            # any value is a dict (it iterates every entry calling .items()).
            if sample is not None:
                sample["scene_3d"] = str(scene_fo3d_path)
                print(
                    f"[_process_video] Set sample['scene_3d'] = {scene_fo3d_path}"
                )

                if self.config.enable_alignment and "text_alignment_embedding" in predictions:
                    emb = (
                        predictions["text_alignment_embedding"]
                        .detach().float().cpu().numpy()
                    )
                    sample["text_alignment_embedding"] = emb.tolist()
                    print(
                        f"[_process_video] Set sample['text_alignment_embedding'] "
                        f"(shape {emb.shape})"
                    )
            else:
                logger.warning(
                    "[_process_video] sample is None — scene_3d cannot be stored "
                    "on the sample directly."
                )

            # Return ONLY frame-level labels.
            # FiftyOne's add_labels() receives this dict and routes each integer
            # key to sample.frames[frame_number] with the nested field dict.
            print(
                f"[_process_video] Done. "
                f"scene_3d → {scene_fo3d_path}  |  "
                f"{n_frames} frame depth maps saved."
            )
            print(f"{'='*60}\n")

            return label_dict

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(
        self,
        video_path: str,
        output_dir: str,
        sample_fps: float,
    ) -> Tuple[List[str], List[int]]:
        """Extract frames from a video file using cv2.

        Args:
            video_path: path to the video file
            output_dir: directory to write frame PNGs
            sample_fps: target frames per second to sample

        Returns:
            (frame_paths, frame_indices): lists of saved PNG paths and the
            corresponding original frame indices (0-based)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"[_extract_frames] Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_fps <= 0:
            logger.warning(
                f"[_extract_frames] Could not read FPS from {video_path}; "
                "assuming 30 fps."
            )
            video_fps = 30.0

        sample_fps = max(float(sample_fps), 0.1)
        frame_interval = max(int(round(video_fps / sample_fps)), 1)

        print(
            f"[_extract_frames] video_fps={video_fps:.2f}  "
            f"total_frames={total_frames}  "
            f"frame_interval={frame_interval}  "
            f"(target sample_fps={sample_fps})"
        )

        frame_paths: List[str] = []
        frame_indices: List[int] = []
        frame_idx = 0
        saved_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_interval == 0:
                out_path = os.path.join(output_dir, f"{saved_idx:06d}.png")
                cv2.imwrite(out_path, frame)
                frame_paths.append(out_path)
                frame_indices.append(frame_idx)
                saved_idx += 1
            frame_idx += 1

        cap.release()
        print(
            f"[_extract_frames] Saved {len(frame_paths)} frames "
            f"(of {total_frames} total) to {output_dir}"
        )
        return frame_paths, frame_indices

    # ------------------------------------------------------------------
    # Depth map visualisation
    # ------------------------------------------------------------------

    def _save_depth_png(self, depth: np.ndarray, output_path: Path) -> None:
        """Normalise a depth map and save as a single-channel grayscale PNG.

        Saves a uint8 single-channel image so that fo.Heatmap(map_path=...,
        range=[0, 255]) renders correctly in the FiftyOne App (FiftyOne applies
        its own colormap overlay on single-channel heatmap images).

        Args:
            depth: raw depth values, shape [H, W]
            output_path: destination file path
        """
        try:
            valid_mask = np.isfinite(depth) & (depth > 0)

            if np.any(valid_mask):
                d_min = np.percentile(depth[valid_mask], 5)
                d_max = np.percentile(depth[valid_mask], 95)
                if d_max > d_min:
                    norm = np.clip(
                        (depth - d_min) / (d_max - d_min), 0.0, 1.0
                    )
                else:
                    norm = np.zeros_like(depth)
            else:
                logger.warning(
                    f"[_save_depth_png] No valid depth values for {output_path}"
                )
                norm = np.zeros_like(depth)

            norm[~valid_mask] = 0.0
            depth_uint8 = (norm * 255).astype(np.uint8)
            cv2.imwrite(str(output_path), depth_uint8)

        except Exception as exc:
            logger.error(
                f"[_save_depth_png] Error saving {output_path}: {exc}",
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Depth unprojection (inlined from demo_gradio.py)
    # ------------------------------------------------------------------

    def _unproject_depth_map_to_point_map(
        self,
        depth: np.ndarray,   # [H, W]
        extrinsic: np.ndarray,  # [3, 4]
        intrinsic: np.ndarray,  # [3, 3]
    ) -> np.ndarray:
        """Back-project a depth map to world-space XYZ coordinates.

        Matches the ``unproject_depth_map_to_point_map`` function in
        ``demo_gradio.py`` exactly, adapted for single-frame (N=1) inputs.

        Returns:
            world_points: [H, W, 3]
        """
        H, W = depth.shape

        y_coords, x_coords = np.meshgrid(
            np.arange(H), np.arange(W), indexing="ij"
        )

        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        camera_points = np.stack(
            [
                (x_coords - cx) / fx * depth,
                (y_coords - cy) / fy * depth,
                depth,
            ],
            axis=-1,
        )  # [H, W, 3]

        R = extrinsic[:3, :3]       # [3, 3]
        T = extrinsic[:3, 3]        # [3,]

        # world = R^T @ (camera - T)
        world_points = np.einsum(
            "ij,hwj->hwi",
            R.T,
            camera_points - T[None, None, :],
        )
        return world_points  # [H, W, 3]

    # ------------------------------------------------------------------
    # Confidence filtering & point cloud accumulation
    # ------------------------------------------------------------------

    def _unproject_and_filter(
        self,
        depth: np.ndarray,       # [H, W]
        conf: np.ndarray,        # [H, W]
        extrinsic: np.ndarray,   # [3, 4]
        intrinsic: np.ndarray,   # [3, 3]
        color_chw: np.ndarray,   # [3, H, W]  values in [0, 1]
        confidence_threshold: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Unproject depth to 3D and filter by confidence.

        Args:
            confidence_threshold: percentile (0–100); points whose confidence
                falls below this percentile are discarded.

        Returns:
            (points [N, 3], colors [N, 3]) or (None, None) if no valid points.
        """
        try:
            world_pts = self._unproject_depth_map_to_point_map(
                depth, extrinsic, intrinsic
            )  # [H, W, 3]

            # Build mask: confidence above threshold AND depth finite & positive
            threshold_val = np.percentile(conf, confidence_threshold)
            valid_depth = np.isfinite(depth) & (depth > 0)
            mask = (conf >= threshold_val) & valid_depth

            if not np.any(mask):
                logger.warning(
                    "[_unproject_and_filter] No points survived filtering."
                )
                return None, None

            points = world_pts[mask]                              # [N, 3]
            color_hwc = color_chw.transpose(1, 2, 0)             # [H, W, 3]
            colors = color_hwc[mask]                              # [N, 3]  [0,1]

            return points.astype(np.float32), colors.astype(np.float32)

        except Exception as exc:
            logger.error(
                f"[_unproject_and_filter] Error: {exc}", exc_info=True
            )
            return None, None

    # ------------------------------------------------------------------
    # Point cloud saving
    # ------------------------------------------------------------------

    def _save_merged_scene(
        self,
        points: np.ndarray,     # [N, 3]
        colors: np.ndarray,     # [N, 3]  values in [0, 1]
        fo3d_path: Path,
    ) -> None:
        """Save the merged multi-frame point cloud as .pcd + .fo3d.

        Args:
            points: world-space XYZ coordinates, shape [N, 3]
            colors: RGB colours in [0, 1], shape [N, 3]
            fo3d_path: destination .fo3d file path (.pcd uses same stem)
        """
        try:
            pcd_path = fo3d_path.with_suffix(".pcd")

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(
                np.clip(colors, 0.0, 1.0)
            )

            o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=False)
            print(
                f"[_save_merged_scene] Saved PCD ({len(points):,} pts) → {pcd_path}"
            )

            scene = fo.Scene()
            scene.camera = fo.PerspectiveCamera(up="Z")
            scene.add(fo.PointCloud("pointcloud", str(pcd_path)))
            scene.write(str(fo3d_path))
            print(f"[_save_merged_scene] Saved fo3d → {fo3d_path}")

        except Exception as exc:
            logger.error(
                f"[_save_merged_scene] Error saving scene: {exc}",
                exc_info=True,
            )

    def _save_empty_scene(self, fo3d_path: Path) -> None:
        """Write a minimal empty fo3d so FiftyOne has a valid file to load."""
        try:
            scene = fo.Scene()
            scene.camera = fo.PerspectiveCamera(up="Z")
            scene.write(str(fo3d_path))
            print(f"[_save_empty_scene] Written empty scene → {fo3d_path}")
        except Exception as exc:
            logger.error(
                f"[_save_empty_scene] Error: {exc}", exc_info=True
            )
