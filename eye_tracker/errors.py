class VideoDeviceUnavailableError(Exception):
    def __init__(self) -> None:
        super().__init__("Could not open video device")


class FrameReadError(Exception):
    def __init__(self) -> None:
        super().__init__("Could not read frame")
