import logging
import os

from huggingface_hub import hf_hub_download
from fiftyone.operators import types

from .zoo import VGGTOmegaModel, VGGTOmegaModelConfig

logger = logging.getLogger(__name__)

# Maps base_name (from manifest) → filename on HuggingFace
_HF_REPO_ID = "facebook/VGGT-Omega"

_FILENAME_MAP = {
    "facebook/VGGT-Omega-1B-512": "vggt_omega_1b_512.pt",
    "facebook/VGGT-Omega-1B-256-Text": "vggt_omega_1b_256_text.pt",
}

_DEFAULT_IMAGE_RESOLUTION = {
    "facebook/VGGT-Omega-1B-512": 512,
    "facebook/VGGT-Omega-1B-256-Text": 256,
}


def download_model(model_name, model_path):
    """Downloads a VGGT-Omega checkpoint from HuggingFace.

    The checkpoint is a raw state dict (.pt file), not a serialized model
    object, so we use hf_hub_download rather than snapshot_download.

    Args:
        model_name: the ``base_name`` declared in the manifest
        model_path: the absolute destination path for the .pt file, as
            declared by the ``base_filename`` field of the manifest
    """
    if model_name not in _FILENAME_MAP:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Supported models: {list(_FILENAME_MAP.keys())}"
        )

    filename = _FILENAME_MAP[model_name]
    dest_dir = os.path.dirname(model_path)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"[VGGT-Omega] Downloading '{filename}' from {_HF_REPO_ID} ...")
    downloaded = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=filename,
        local_dir=dest_dir,
    )
    print(f"[VGGT-Omega] Downloaded to: {downloaded}")

    # hf_hub_download writes to {dest_dir}/{filename}; if model_path differs
    # from that location, rename to match what FiftyOne expects.
    expected = os.path.join(dest_dir, filename)
    if os.path.abspath(downloaded) != os.path.abspath(model_path):
        import shutil
        shutil.move(downloaded, model_path)
        print(f"[VGGT-Omega] Moved checkpoint to: {model_path}")


def load_model(
    model_name=None,
    model_path=None,
    confidence_threshold=50.0,
    video_sample_fps=2.0,
    image_resolution=None,
    enable_alignment=False,
    **kwargs,
):
    """Load a VGGT-Omega model for use with FiftyOne video datasets.

    Args:
        model_name: the ``base_name`` from the manifest (e.g.
            ``"facebook/VGGT-Omega-1B-512"``)
        model_path: absolute path to the downloaded .pt checkpoint file
        confidence_threshold (float): percentile (0–100) used to filter
            low-confidence points from the merged point cloud. Default 50.0.
        video_sample_fps (float): frames per second to sample from the input
            video. Default 2.0. Increase for denser temporal coverage at the
            cost of GPU memory and runtime.
        image_resolution (int): target resolution for VGGT-Omega's patch
            tokeniser. Must be 512 for the 1B-512 checkpoint and 256 for the
            1B-256-Text checkpoint. Defaults to the per-model recommended
            value when not specified.
        enable_alignment (bool): if True, activates the TextAlignmentHead and
            stores a 2048-D L2-normalised scene embedding on each sample.
            Only meaningful with the 1B-256-Text checkpoint. Default False.
        **kwargs: forwarded verbatim into VGGTOmegaModelConfig

    Returns:
        VGGTOmegaModel: model ready for ``dataset.apply_model(model, field)``
    """
    if image_resolution is None:
        image_resolution = _DEFAULT_IMAGE_RESOLUTION.get(model_name, 512)
        print(
            f"[VGGT-Omega] image_resolution not set; "
            f"using {image_resolution} for '{model_name}'"
        )

    config_dict = {
        "model_name": model_name,
        "model_path": model_path,
        "confidence_threshold": confidence_threshold,
        "video_sample_fps": video_sample_fps,
        "image_resolution": image_resolution,
        "enable_alignment": enable_alignment,
    }
    config_dict.update(kwargs)

    print(f"[VGGT-Omega] Building model config: {config_dict}")
    config = VGGTOmegaModelConfig(config_dict)
    return VGGTOmegaModel(config)


def resolve_input(model_name, ctx):
    """Defines the operator panel UI for VGGT-Omega parameters.

    Args:
        model_name: the name of the model
        ctx: an ExecutionContext

    Returns:
        a fiftyone.operators.types.Property
    """
    inputs = types.Object()

    inputs.float(
        "video_sample_fps",
        default=2.0,
        label="Video Sample FPS",
        description=(
            "Frames per second to sample from the input video. "
            "Higher values → denser temporal coverage but more GPU memory."
        ),
    )

    inputs.float(
        "confidence_threshold",
        default=50.0,
        label="Confidence Threshold (%)",
        description=(
            "Percentile (0–100) used to filter low-confidence depth points "
            "from the merged 3D point cloud. Higher → fewer, more reliable points."
        ),
    )

    default_res = _DEFAULT_IMAGE_RESOLUTION.get(model_name, 512)
    inputs.int(
        "image_resolution",
        default=default_res,
        label="Image Resolution",
        description=(
            "Target resolution for VGGT-Omega's tokeniser. "
            "Use 512 for the 1B-512 checkpoint, 256 for 1B-256-Text."
        ),
    )

    if model_name == "facebook/VGGT-Omega-1B-256-Text":
        inputs.bool(
            "enable_alignment",
            default=False,
            label="Enable Text Alignment Embedding",
            description=(
                "If True, computes and stores a 2048-D scene embedding "
                "suitable for similarity search. Only available on the "
                "256-Text checkpoint."
            ),
        )

    return types.Property(inputs)
