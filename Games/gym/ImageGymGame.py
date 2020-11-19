import typing

import numpy as np
import cv2

from utils.game_utils import GymState
from .GymGame import GymGame


class ImageGymGame(GymGame):

    def __init__(self, env_name: str) -> None:
        super().__init__(env_name)
        self.x, self.y, self.depth = 96, 96, 3

    def getDimensions(self, **kwargs) -> typing.Tuple[int, ...]:
        return self.x, self.y, self.depth

    def buildObservation(self, state: GymState, **kwargs) -> np.ndarray:
        im = state.env.render(mode='rgb_array')  # Note this also creates a live display (no workaround yet).
        resized = cv2.resize(im, (self.x, self.y), interpolation=cv2.INTER_AREA)
        return resized
