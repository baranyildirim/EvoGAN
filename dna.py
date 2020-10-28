from parameters import Parameters

class DNA:
    def __init__(self, parameters: Parameters):
        self.parameters = parameters
        return

    def serialize(self, Parameters) -> int:
        return self.parameters.serialize()

    def mutate(self):
        pass

    def evolve(self):
        pass
