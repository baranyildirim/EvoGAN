from parameters import Parameters, FirstCellParameters, SecondCellParameters, ThirdCellParameters
from cell_dna import DNAProperties, CellDNA
from dataclasses import dataclass, astuple, fields, asdict
from typing import Type, TypeVar, List, Callable, Any, Sequence, Tuple
from random import choice, uniform
import numpy as np  

D = TypeVar('D', bound='DNA')
class DNA:
    def __init__(self, parameters: List[Parameters], properties: DNAProperties = DNAProperties()):
        self.cells = []
        for param in parameters:
            self.cells.append(CellDNA(param, properties))
        self.properties = properties
        return

    def __repr__(self):
        r = ""
        for c in self.cells:
            r += repr(c)
        return r

    @classmethod
    def gen_random(
        cls:D, 
        param_types: List[Type[Parameters]] = [FirstCellParameters, SecondCellParameters, ThirdCellParameters], 
        prop: DNAProperties = DNAProperties()
    ) -> D:         
        params = [] 
        for param_type in param_types:
            params.append(param_type.gen_random())
        return DNA(params, prop)

    def serialize(self) -> List[int]:
        s = []
        for c in self.cells:
            s += c.serialize()
        return s

    def mutate(self) -> None:
        for c in self.cells:
            c.mutate()
        return

    def set_properties(self, properties: DNAProperties) -> None:
        self.properties = properties
        for c in self.cells:
            c.set_properties(properties)
        return

    def evolve(self, evolution_matrix: List[List[float]]):
        position = 0
        for c in self.cells:
            param_count = len(c.serialize())
            c.evolve(
                evolution_matrix[position:param_count]
            )
            position = param_count
        return

    def to_arch(self):
        return