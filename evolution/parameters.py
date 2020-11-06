from enum import Enum, IntEnum
from dataclasses import dataclass, astuple, fields, asdict
from typing import Type, TypeVar, List, Any
from random import choice

class Skip(IntEnum):
    NO_SKIP_CONNECTION = 0
    SKIP_CONNECTION = 1

class ConvolutionBlock(IntEnum):
    PRE_ACTIVATION = 0
    POST_ACTIVATION = 1

class Normalization(IntEnum):
    NO_NORMALIZATION = 0
    BATCH_NORMALIZATION = 1
    INSTANCE_NORMALIZATION = 2

class Upsample(IntEnum):
    BILINEAR = 0
    NEAREST_NEIGHBOR = 1
    DECONV = 2

class Shortcut(IntEnum):
    NO_SHORTCUT = 0
    SHORTCUT = 1

P = TypeVar('P', bound='Parameters')

@dataclass 
class Parameters: 
    @classmethod
    def gen_random(cls: Type[P], choice_func=choice) -> P:
        field_list = fields(cls)
        params = []
        for f in field_list:
            param_choice = choice_func(list(f.type.__members__))
            param_choice = f.type[param_choice]
            params.append(param_choice)
        return cls._from_serial(params)

    @classmethod
    def from_serial(cls: Type[P], s: List[int]) -> P:
        field_list = fields(cls)
        if (len(field_list) != len(s)):
            raise Exception(f"Bad field length:{len(s)} should be: {len(field_list)}")
        params = []
        for idx, f in enumerate(field_list):
            param = f.type[list(f.type.__members__)[s[idx]]]
            params.append(param)
        return cls._from_serial(params)
            
    @classmethod
    def _from_serial(cls: Type[P], s: List[int]) -> P:
        return cls(*s)
    
    @classmethod
    def get_field_options(cls: Type[P], field_idx:int) -> List[Any]:
        field_list = fields(cls)
        if (field_idx >= len(field_list)):
            raise Exception(f"Field at idx: {field_idx} does not exist.")
        return list(field_list[field_idx].type)

    @classmethod
    def parameter_count(cls: Type[P]) -> int:
        return len(fields(cls))

    def serialize(self) -> List[int]:
        values = astuple(self)
        values = list(values)
        for idx, v in enumerate(values):
            values[idx] = v.value
        return values

    def to_dict(self) -> dict:
        values = asdict(self)
        return values 

@dataclass
class FirstCellParameters(Parameters):
    conv_block : ConvolutionBlock
    normalization : Normalization
    upsample: Upsample
    shortcut : Shortcut

@dataclass
class SecondCellParameters(Parameters):
    conv_block : ConvolutionBlock
    normalization : Normalization
    upsample: Upsample 
    shortcut : Shortcut
    skip_from_1: Skip

@dataclass
class ThirdCellParameters(Parameters):
    conv_block : ConvolutionBlock
    normalization : Normalization
    upsample: Upsample
    shortcut : Shortcut 
    skip_from_1: Skip
    skip_from_2: Skip

    # Merge skips while serializing
    def serialize(self) -> List[int]:
        values = Parameters.serialize(self)[:self.parameter_count() - 2]
        merged_skip = int(str(self.skip_from_1.value) + str(self.skip_from_2.value))
        values.append(merged_skip)
        return values

    # Unmerge skips while de-serializing
    @classmethod
    def from_serial(cls: Type[P], s: List[int]) -> P:
        merged_skip = s[-1]
        param = s.copy()[:cls.parameter_count() - 2]
        if merged_skip == 0:
            param.extend([0, 0])
        if merged_skip == 1:
            param.extend([0, 1])
        if merged_skip == 11:
            param.extend([1, 1])
        if merged_skip == 10:
            param.extend([1, 0])
        return super().from_serial(param)
