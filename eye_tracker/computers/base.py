from abc import ABC, abstractmethod

import numpy as np


class ComputePipeline(ABC):
    @abstractmethod
    def compute(self, frame: np.typing.NDArray[np.uint8]) -> np.typing.NDArray[np.uint8]:
        """Process the input frame and return a frame of the same shape."""
        pass
