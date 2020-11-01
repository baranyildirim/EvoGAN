#!/usr/bin/env python3
import numpy as np


def generate_new_dna(n_dna, dna_length, p_dists=None):

    if p_dists is None:
        # Make distributions uniform
        p_dists = 0.5 * np.ones((dna_length, 2))

    dna_list = []

    for i in range(dna_length):

        dna = np.random.choice([0, 1], size=(n_dna, 1), p=p_dists[i])
        dna_list.append(dna)

    return np.concatenate(dna_list, axis=1)

# This will create GAN from dna
# train GAN and
# return Inception score
def score_dna(dna):
    # This is the target dna
    target_dna = np.array([1, 0, 0, 1, 0])

    # This is the reward for matching each position
    reward_dict = np.array([3.0, 8.0, 4.0, 2.0, 5.0])

    # Find matches between dna and target dna
    matches = 1 * (dna == target_dna)

    # Compute rewards for dna
    rewards = np.sum(matches * reward_dict, axis=1)

    return rewards


def generation_step(dna, scores, mut_prob=0):

    n_dna, dna_length = dna.shape

    p_dist = np.zeros((dna_length, 2))

    # Compute probabilities using scores

    for n in range(dna_length):
        p0 = np.sum(scores[dna[:, n] == 0])
        p1 = np.sum(scores[dna[:, n] == 1])
        z = p0 + p1
        p0 /= z
        p1 /= z

        p_dist[n, 0] = p0
        p_dist[n, 1] = p1

    print(p_dist)
    dna = generate_new_dna(n_dna, dna_length, p_dist)

    # Mutation
    mutation = np.random.choice([0, 1], size=dna.shape, p=[1-mut_prob, mut_prob])
    dna = 1 * np.logical_xor(dna, mutation)

    return dna


def main():
    # Initialize dna uniformly
    dna = generate_new_dna(10, 5)


    # Initialize mutation probability
    mut_prob = 0.05

    n_epochs = 20

    for epoch in range(n_epochs):
        rewards = score_dna(dna)
        print(dna, rewards)
        dna = generation_step(dna, rewards, mut_prob)

        if epoch % 2 == 0:
            mut_prob /= 2


if __name__ == "__main__":
    main()