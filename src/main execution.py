# =========================
# Main Execution
# =========================

import os
from pathlib import Path

from CameraLoader import CameraConfigLoader
from Factories import AreaCalculatorFactory, AreaMethod, SegmentationMethod, SegmenterFactory
from ObjectFilterStrategy import SimpleObjectFilter
from OutputWriters import CSVOutputWriter
from VideoProcessorPipeline import VideoProcessor

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_VIDEOS_DIR = SCRIPT_DIR.parent / "test_videos"
RESULTS_DIR = SCRIPT_DIR.parent / "results"


def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    loader = CameraConfigLoader()

    cam = loader.get_camera_info(
        camera_id="cam1",
        video_path=str(TEST_VIDEOS_DIR / "cam1.mp4"),
        video_save_path=str(RESULTS_DIR / "cam1_output.mp4"),
        camera_config="cameras_config.json"
    )

    seg_factory = SegmenterFactory()
    area_factory = AreaCalculatorFactory()

    segmenter = seg_factory.create(
        SegmentationMethod.CLASSICAL
    )

    obj_filter = SimpleObjectFilter(
        target_labels=["Broiler"],
        template_mask="broiler_template.png",
        min_area=500.0
    )

    area_calc = area_factory.create(
        AreaMethod.CALIBRATED,
        pixel_to_cm=cam.pixel_to_cm
    )

    output_writer = CSVOutputWriter(csv_path=str(RESULTS_DIR / f"{cam.camera_id}_results.csv"))

    processor = VideoProcessor(
        camera_config=cam,
        segmenter=segmenter,
        area_calculator=area_calc,
        obj_filter=obj_filter,
        output_writer=output_writer,
        max_frames=None
    )
    processor.run()


if __name__ == "__main__":
    main()
