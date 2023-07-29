from .encoder import *
from .block_encoder import *
from .convolutional_encoder import *
from .trellis import *
from .encoder_registry import *
from .turbo import *
from .turboae import *

from typing import Union

SizedEncoder = Union[TurboEncoder, TrellisEncoder, BlockEncoder, ENC_interCNN]
