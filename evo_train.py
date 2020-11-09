#!/usr/bin/env python3

import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'AutoGAN'))

import numpy as np
from typing import List
from evolution.dna import DNA
from evolution.cell_dna import DNAProperties
from gan_train import train_gan

def generate_new_dna(n_dna:int, properties: DNAProperties) -> List[DNA]:
    dna_list = []
    for i in range(n_dna):
        dna_list.append(DNA.gen_random())
    return dna_list

def output_dna(dna_list: List[DNA]) -> None:
    """ Print serialized form of a list of DNAs """
    for d in dna_list:
        print(d.serialize())

def score_dna(dna: DNA) -> float:
    """ Create a GAN using the DNA,
        train the GAN and return the inception score.
        Training uses train_derived from AutoGAN
    """
    reward = train_gan(to_arch(dna), 1)
    return reward

def scoring_step(dna_list: List[DNA]) -> List[float]:
    """ Score each DNA """
    scores = []
    for d in dna_list:
        print(to_arch(d))
        scores.append(score_dna(d))
    return scores

def generation_step(dna_list: List[DNA], scores: List[List[float]], properties: DNAProperties) -> List[DNA]:
    """ Set new properties (usually mutation probability)
        Evolve and mutate a list of DNAs and return the list
    """
    for d in dna_list:
        d.set_properties(properties)
    
    # Evoluton:
    evo_matrix = generate_evolution_matrix(dna_list, scores)
    for d in dna_list:
        d.evolve(evo_matrix)

    # Mutation
    for d in dna_list:
        d.mutate()

    return dna_list

def to_arch(dna: DNA) -> List[int]:
    """ Convert DNA to architecture string
        expected by AutoGAN trainer
    """
    values = dna.serialize()   
    skip = values[len(values)-2:]
    skip = skip[0] * 2 + skip[1]
    values = values[:len(values)-2]
    values.append(skip)
    return values

def generate_evolution_matrix(dna_list: List[DNA], scores: List[int]) -> List[List[float]]:
    """ Generate a probability distribution of evolutopn parameters
        based on inception scores from the scoring step
    """
    param_options = []
    evo_matrix = []

    for c in dna_list[0].cells:
        for idx, _ in enumerate(c.serialize()):
            param_options.append(len(c.parameters.get_field_options(idx)))

    for i in range(len(dna_list[0].serialize())):
        p = [0 for _ in range(param_options[i])]
        for d_idx, d in enumerate(dna_list):
            p[d.serialize()[i]] += scores[d_idx]
        p_sum = sum(p)
        p = [x / p_sum for x in p]
        evo_matrix.append(p)

    return evo_matrix
    

def main():
    # Initialize mutation probability
    mut_prob = 0.05
    properties = DNAProperties(mutation_probability=mut_prob)

    # Initialize dna uniformly
    dna_list = generate_new_dna(10, properties)
    n_epochs = 20

    for epoch in range(n_epochs):
        properties = DNAProperties(mutation_probability=mut_prob)
        print(f"\n EPOCH: {epoch}\n")
        rewards = scoring_step(dna_list)
        dna_list = generation_step(dna_list, rewards, properties)
        output_dna(dna_list)

        if epoch % 2 == 0:
            mut_prob /= 2

if __name__ == "__main__":
    main()
