#!/usr/bin/env python3

import os
import sys

import numpy as np
import logging
import torch
import multiprocessing as mp

from typing import List
from evolution.dna import DNA
from evolution.cell_dna import DNAProperties
from gan_train import train_gan

evo_train_logger = logging.getLogger("evo_train")

def init_logger():
    """ Initalize evo_train_logger """
    evo_train_logger.setLevel(logging.INFO)
    fh = logging.FileHandler('evo_train.log')
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    evo_train_logger.addHandler(fh)
    evo_train_logger.addHandler(ch)


def generate_new_dna(n_dna:int, properties: DNAProperties) -> List[DNA]:
    dna_list = []
    for i in range(n_dna):
        dna_list.append(DNA.gen_random())
    return dna_list

def output_dna(dna_list: List[DNA], scores:List[float]) -> None:
    """ Print serialized form of a list of DNAs 
        with their respective inception scores
    """
    assert(len(dna_list) == len(scores))
    for idx, d in enumerate(dna_list):
        print(f"{d.serialize()} : {scores[idx]}")
        evo_train_logger.info(f"{d.serialize()} : {scores[idx]}")




def score_dna(dna: DNA) -> float:
    """ Create a GAN using the DNA,
        train the GAN and return the inception score.
        Training uses train_derived from AutoGAN
    """
    reward = train_gan(to_arch(dna), 1)
    return reward

def scoring_step(dna_list: List[DNA]) -> List[float]:
    """ Score each DNA """

    gpu_count = torch.cuda.device_count()
    num_processes = gpu_count
    pool = mp.Pool(num_processes)

    scores = []
    for d_idx in range(0, len(dna_list) // num_processes, num_processes):
        d_args = dna_list[d_idx * num_processes:d_idx * num_processes + num_processes]
        curr_scores = [pool.apply(score_dna, args=d_elem) for d_elem in d_args]
        scores.extend(curr_scores)

    pool.close()  
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
        evo_train_logger.info(f"\n EPOCH: {epoch}\n")
        inception_scores = scoring_step(dna_list)
        output_dna(dna_list, inception_scores)

        dna_list = generation_step(dna_list, inception_scores, properties)

        if epoch % 2 == 0:
            mut_prob /= 2

    final_dna_list = dna_list
    final_scores = scoring_step(final_dna_list)

    print("\n FINAL: \n")
    evo_train_logger.info("\n FINAL: \n")
    output_dna(final_dna_list, final_scores)
    

if __name__ == "__main__":
    init_logger()
    main()
