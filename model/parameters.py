from enum import Enum
from dataclasses import dataclass, astuple, fields, asdict
from typing import Type, TypeVar, List, Any
from random import choice

# we need separate skip connection feature
# coming from each layer
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

class Upsample(Enum):
    BILINEAR = 0
    NEAREST_NEIGHBOR = 1
    STRIDE_2 = 2

class Shortcut(Enum):
    NO_SHORTCUT = 0
    SHORTCUT = 1

P = TypeVar('P', bound='Parameters')

@dataclass 
class Parameters: 
    skip : Skip
    conv_block : ConvolutionBlock
    normalization : Normalization
    upsample: Upsample
    shortcut : Shortcut

    @classmethod
    def gen_random(cls: Type[P], choice_func=choice) -> P:
        field_list = fields(cls)
        params = []
        for f in field_list:
            param_choice = choice_func(list(f.type.__members__))
            params.append(param_choice)
        return Parameters.from_serial(params)
            
    @classmethod
    def from_serial(cls: Type[P], s: List[int]) -> P:
        return Parameters(*s)

    @staticmethod
    def get_field_options(field_idx:int) -> List[Any]:
        field_list = fields(Parameters)
        if (field_idx >= len(field_list)):
            raise Exception(f"Field at idx: {field_idx} does not exist.")
        return list(field_list[field_idx].type.__members__)

    def serialize(self) -> List[int]:
        values = astuple(self)
        return list(values)

    def to_dict(self) -> dict:
        values = asdict(self)
        return values 
