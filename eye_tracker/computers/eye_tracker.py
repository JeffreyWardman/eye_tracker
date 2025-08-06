import cv2
import dlib
import numpy as np
from pydantic import BaseModel

from . import ComputePipeline


class Landmarks(BaseModel):
    top: int
    bottom: int
    left: int
    right: int


class Bounds(BaseModel):
    """A class to represent the bounds of an eye in a video frame."""

    x_min: int
    x_max: int
    y_min: int
    y_max: int

    @staticmethod
    def from_points(points: tuple) -> "Bounds":
        """Set bounds from a list of dlib points."""
        xs = [p.x for p in points]
        ys = [p.y for p in points]
        return Bounds(x_min=int(min(xs)), x_max=int(max(xs)), y_min=int(min(ys)), y_max=int(max(ys)))


# Assuming facing the person
LEFT_EYE_LANDMARKS = Landmarks(top=37, bottom=41, left=36, right=39)
RIGHT_EYE_LANDMARKS = Landmarks(top=43, bottom=47, left=42, right=45)


class EyeTracker(ComputePipeline):
    def __init__(
        self,
        predictor_path: str = "assets/shape_predictor_68_face_landmarks.dat",
        logo_path: str = "assets/oakton_logo.jpeg",
        debug: bool = False,
    ) -> None:
        super().__init__()

        self.debug = debug

        # Initialize the face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        # Load logo
        self.logo = cv2.imread(logo_path)

    def compute(self, frame: np.typing.NDArray[np.uint8]) -> np.typing.NDArray[np.uint8]:
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        rects = self.detector(frame_grayscale, 0)

        # Loop over the face detections
        for rect in rects:
            shape = self.predictor(frame_grayscale, rect)

            left_eye = Bounds.from_points((
                shape.part(LEFT_EYE_LANDMARKS.left),
                shape.part(LEFT_EYE_LANDMARKS.right),
                shape.part(LEFT_EYE_LANDMARKS.top),
                shape.part(LEFT_EYE_LANDMARKS.bottom),
            ))

            right_eye = Bounds.from_points((
                shape.part(RIGHT_EYE_LANDMARKS.left),
                shape.part(RIGHT_EYE_LANDMARKS.right),
                shape.part(RIGHT_EYE_LANDMARKS.top),
                shape.part(RIGHT_EYE_LANDMARKS.bottom),
            ))

            # Draw the logo on the eyes
            frame = self._draw_logo(frame, left_eye, inflate_amount=20)
            frame = self._draw_logo(frame, right_eye, inflate_amount=20)

        return frame

    def _draw_logo(self, frame: np.typing.NDArray[np.uint8], eye: Bounds, inflate_amount: int) -> np.typing.NDArray[np.uint8]:
        half = int(inflate_amount / 2)

        eye_height = eye.y_max - eye.y_min + inflate_amount
        eye_width = eye.x_max - eye.x_min + inflate_amount

        if eye_height <= 0 or eye_width <= 0:
            # Eye dimensions are invalid. Return the frame unchanged
            return frame

        if self.debug:
            # Optionally, draw the eye rectangle for debugging
            cv2.rectangle(frame, (eye.x_min - half, eye.y_min - half), (eye.x_max + half, eye.y_max + half), (0, 255, 0), 1)

            # Draw the eye landmarks for debugging
            cv2.circle(frame, (eye.x_min, eye.y_min), 5, (255, 0, 0), -1)  # Top left: Blue
            cv2.circle(frame, (eye.x_max, eye.y_min), 5, (0, 255, 0), -1)  # Top right: Green
            cv2.circle(frame, (eye.x_min, eye.y_max), 5, (0, 0, 255), -1)  # Bottom left: Red
            cv2.circle(frame, (eye.x_max, eye.y_max), 5, (255, 255, 0), -1)  # Bottom right: Cyan
            return frame

        # Resize logo to fit the eye
        logo_resized = cv2.resize(self.logo, (eye_width, eye_height), interpolation=cv2.INTER_AREA)

        # Place the logo on the frame
        frame[eye.y_min - half : eye.y_max + half, eye.x_min - half : eye.x_max + half] = logo_resized
        return frame
