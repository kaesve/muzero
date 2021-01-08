"""
Import DotDict within the utils global namespace.
Important: Imports the entire debugging namespace to ensure that tensorflow does not hog all VRAM within
           a main session on the GPU.
"""
from .debugging import *  # Required to initialize tensorflow (failed to get CUDNN Handle exception)
from .storage import DotDict
