# =========================
# Camera Config & Loader
# =========================

from dataclasses import dataclass
from pathlib import Path


@dataclass
class CameraConfig:
    camera_id: str
    video_path: Path
    video_save_path: Path


class CameraConfigLoader:

    def get_camera_info(self, camera_id: str, video_path: str, video_save_path: str) -> CameraConfig:
        return CameraConfig(
            camera_id=camera_id.lower(),
            video_path=Path(video_path),
            video_save_path=Path(video_save_path),
        )
