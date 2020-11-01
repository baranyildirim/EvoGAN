from parameters import Parameters
from dataclasses import dataclass, astuple, fields, asdict
from typing import Type, TypeVar, List, Callable, Any, Sequence, Tuple
from random import choice, uniform
import numpy as np  

D = TypeVar('D', bound='DNA')

@dataclass
class DNAProperties:
    choice_func = choice
    mutation_probability : int = 0.2
    mutation_probability_func = uniform
    mutation_probability_args : Tuple = (0, 1)

class DNAParameters(Parameters):
    

class DNA:
    def __init__(self, parameters: Parameters, properties: DNAProperties = DNAProperties()):
        self.parameters = parameters
        self.properties = properties
        self.parameter_history : List[dict] = []
        self._append_history()
        return

    def _append_history(self):
        self.parameter_history.append(self.parameters.to_dict())

    def __repr__(self):
        return repr(self.parameters.to_dict())

    @classmethod
    def gen_random(cls:D, prop: DNAProperties = DNAProperties()) -> D:
        return DNA(Parameters.gen_random(choice_func=prop.choice_func))

    def serialize(self) -> List[int]:
        return self.parameters.serialize()

    def mutate(self) -> None:
        current_parameters = self.serialize()
        for idx, p in enumerate(current_parameters):
            rand = self.properties.mutation_probability_func(*self.properties.mutation_probability_args)
            possible_choices = Parameters.get_field_options(idx)
            possible_choices.remove(p)
            if (rand < self.properties.mutation_probability):
                current_parameters[idx] = self.properties.choice_func(possible_choices)

        self.parameters = Parameters.from_serial(current_parameters)
        self._append_history()
        return

    def set_properties(self, properties: DNAProperties) -> None:
        self.properties = properties
        return

    def evolve(self, evolution_matrix: List[List[float]]):
        current_parameters = self.serialize()
        for idx, p in enumerate(current_parameters):
            possible_choices = Parameters.get_field_options(idx)
            current_parameters[idx] = np.random.choice(possible_choices, p=evolution_matrix[idx])
        self.parameters = Parameters.from_serial(current_parameters)
        self._append_history()
        return

    def get_evolution_history(self) -> List[Parameters]:
        return self.parameter_history.copy()

