#!/usr/bin/env python3
import numpy as np
from typing import List
from evolution.dna import DNA
from evolution.cell_dna import DNAProperties

def generate_new_dna(n_dna:int, properties: DNAProperties) -> List[DNA]:
    dna_list = []
    for i in range(n_dna):
        dna_list.append(DNA.gen_random())
    return dna_list


target_dna = DNA.gen_random()
print(f"Target DNA: {target_dna.serialize()}")

def output_dna(dna_list: List[DNA]) -> None:
    for d in dna_list:
        print(d.serialize())

def score_dna(dna: DNA) -> float:
    """ Create a GAN using the DNA,
        train the GAN and return the inception score
    """
    # This is the target dna
    target = np.array(target_dna.serialize())
    actual = np.array(dna.serialize())
    # This is the reward for matching each position
    reward_dict = np.array([
        3.0, 8.0, 4.0, 2.0, 5.0, 
        2.0, 1.0, 5.0, 2.0, 1.0,
        3.0, 8.0, 4.0, 2.0, 1.0
    ])

    # Find matches between dna and target dna
    matches = 1 * (actual == target)

    # Compute rewards for dna
    rewards = np.sum(matches * reward_dict)
    return rewards

def scoring_step(dna_list: List[DNA]) -> List[float]:
    """ Score each DNA """
    scores = []
    for d in dna_list:
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
    properties = DNAProperties(mutation_probability=0.05)

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