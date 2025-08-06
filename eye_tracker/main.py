from typing import cast

import cv2
import numpy as np

from .computers.eye_tracker import ComputePipeline, EyeTracker
from .errors import FrameReadError, VideoDeviceUnavailableError


class VideoStreamPipeline:
    def __init__(self, frame_width: int = 1280, frame_height: int = 960) -> None:
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            raise VideoDeviceUnavailableError()

        # Set frame width and height
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    def __call__(self, computer: ComputePipeline) -> None:
        self.start(computer)

    def start(self, computer: ComputePipeline) -> None:
        while True:
            frame = self.frame()
            processed_frame = computer.compute(frame)
            cv2.imshow("Video Stream", processed_frame)

            if self.exit_check():
                break

        self.stop()

    def frame(self) -> np.typing.NDArray[np.uint8]:
        ret, frame = self.capture.read()
        if not ret:
            raise FrameReadError()
        return cast(np.typing.NDArray[np.uint8], frame)

    def exit_check(self) -> bool:
        key = cv2.waitKey(1) & 0xFF
        return key == ord("q")

    def stop(self) -> None:
        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    from eye_tracker.utils import download_shape_predictor

    download_shape_predictor()

    pipeline = VideoStreamPipeline()
    eye_tracker = EyeTracker(debug=False)
    pipeline.start(eye_tracker)
