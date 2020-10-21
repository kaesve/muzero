# Bug fixing TF2?
# Prevent TF2 from hogging all the available VRAM when initializing?
# @url: https://github.com/tensorflow/tensorflow/issues/24496#issuecomment-464909727
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import logging
import tensorflow as tf

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Bug fixing TF2?

# Suppress warnings from TENSORFLOW's side
logger = tf.get_logger()
logger.setLevel(logging.ERROR)
