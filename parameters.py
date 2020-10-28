from enum import Enum
from dataclasses import dataclass, astuple

class Skip(Enum):
    NO_SKIP_CONNECTION = 0
    SKIP_CONNECTION = 1

class ConvolutionBlock(Enum):
    PRE_ACTIVATION = 0
    POST_ACTIVATION = 1

class Normalization(Enum):
    NO_NORMALIZATION = 0
    BATCH_NORMALIZATION = 1
    INSTANCE_NORMALIZATION = 2

class Shortcut(Enum):
    NO_SHORTCUT = 0
    SHORTCUT = 1

@dataclass 
class Parameters: 
    skip : Skip
    conv_block : ConvolutionBlock
    normalization : Normalization
    shortcut : Shortcut

    def serialize(self) -> int:
        values = astuple(self)
        return int(''.join(str(x) for x in values))

