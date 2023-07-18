from tkinter import W
import numpy as np
import torch
import pickle
from matplotlib import pyplot as plt
from scipy.special import softmax
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from simulation import runCellularAutomata, centerSeedInitial, randomInitial, display_frame


NUM_ITERS = 1000
NUM_SIM_STEPS_PER_ITER = 20
POPULATION_SIZE = 1000
MUTATION_RATE = 0.85

USE_INTEGERS_ORGANISMS = True
USE_INCREMENTAL_MUTATIONS = True
USE_ROULETTE_SELECTION = True
SURVIVAL_FACTOR = 0.25 # for truncation selection
USE_ANNEALING = True

NUM_FEATURES = 13
DISPLAY_GENERATIONS = 10


def make_kernel(organism):
    return np.array([organism[:3],
        [organism[3], 0, organism[4]],
        organism[5:8]
    ])


def make_rule(organism):
    life_to_death_rule = lambda prev_state, kernel_sum: (prev_state == 1) & ( (kernel_sum < organism[8]) | (kernel_sum > organism[9]) )
    life_to_life_rule = lambda prev_state, kernel_sum: ((prev_state == 1) & ((kernel_sum >= organism[10]) & (kernel_sum <= organism[11]))) 
    # Note to self: you can't have a Conway-like == rule if you're using real-valued organisms because it will never be true --
    # if you're using real-valued organisms you need range rules like (<= x & >= y). Using (<= x-0.5 & >= x+0.5) will be equivalent to == for integer
    # organisms and compatible with real-valued organisms.
    death_to_life_rule = lambda prev_state, kernel_sum: ((prev_state == 0) & (kernel_sum >= organism[12] - 0.5) & (kernel_sum <= organism[12] + 0.5))
    return lambda prev_state, kernel_sum: life_to_life_rule(prev_state, kernel_sum) | death_to_life_rule(prev_state, kernel_sum) & ~life_to_death_rule(prev_state, kernel_sum) 


def conway_organism():
    # Any live cell with fewer than two live neighbours dies, as if by underpopulation.
    # Any live cell with two or three live neighbours lives on to the next generation.
    # Any live cell with more than three live neighbours dies, as if by overpopulation.
    # Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
    # Conway: lambda last, k: (((k == 2) | (k == 3)) & (last == 1)) | ((k == 3) & (last == 0))
    return [1] * 8 + [2, 3, 2, 3, 3]


def make_kernels(population):
    return np.stack([make_kernel(organism) for organism in population])


def make_rules(population):
    return [make_rule(organism) for organism in population]


def entropy(state):
    c, b, h, w = state.shape
    total_values = h*w
    total_1s = state.sum(axis=(0, 2, 3))
    p = (total_1s / total_values)
    entropy = -p * torch.log2(p) - (1-p) * torch.log2(1-p)
    entropy[(p==0)|(p==1)] = 0
    return entropy.numpy()


def convolved_entropy(state, score_kernel, stride=None):
    kW, kH = score_kernel.size()[-2:]
    convolved_sums = torch.nn.functional.conv2d(state.transpose(0, 1), score_kernel, stride=kW if stride is None else stride)
    convolved_probs = convolved_sums / (kW*kH)
    convolved_entropies = -convolved_probs * torch.log2(convolved_probs) - (1-convolved_probs) * torch.log2(1-convolved_probs)
    convolved_entropies[(convolved_probs==0)|(convolved_probs==1)] = 0
    convolved_entropies_mean = convolved_entropies.squeeze(1).mean((1, 2)).numpy()
    convolved_entropies_mean[np.isnan(convolved_entropies_mean)] = 0
    # assert np.all((convolved_entropies_mean <= 1) & (convolved_entropies_mean >= 0)), "Entropy score not normalized."
    return convolved_entropies_mean


def sum_score(state):
    return state.sum((0, 2, 3)).numpy()


def sum_score_normalized(state):
    c, b, h, w = state.shape
    total_values = h*w
    total_1s = state.sum(axis=(0, 2, 3))
    total_1s_normalized = (total_1s / total_values).numpy()
    # assert np.all((total_1s_normalized <= 1) & (total_1s_normalized >= 0)), "Sum score not normalized."
    return total_1s_normalized


score_fn_desc = "convolved_entropy_small"
score_kernel_small = torch.Tensor( np.ones((5, 5)) ).unsqueeze(0).unsqueeze(1).detach()
score_kernel_big = torch.Tensor( np.ones((20, 20)) ).unsqueeze(0).unsqueeze(1).detach()
def score(state):
    return convolved_entropy(state, score_kernel_small)


def random_real_organism():
    organism = np.zeros(NUM_FEATURES)
    organism[0:8] = (np.random.rand(8) * 2) - 1 
    organism[8:] = (np.random.rand(5) * 16) - 8 
    # The death lower threshold has to be lower than the death upper threshold.
    while organism[8] > organism[9]: organism[8:10] = (np.random.rand(2) * 16) - 8
    # The life lower threshold has to be lower than the life upper threshld.
    while organism[10] > organism[11]: organism[10:12] = (np.random.rand(2) * 16) - 8 
    # Need at least 1 neighbor to spawn.
    while abs(organism[-1]) <= 0.5: organism[-1] = (np.random.rand(1) * 16) - 8 
    return organism 


def random_integer_organism():
    organism = np.zeros(NUM_FEATURES)
    organism[0:8] = np.random.randint(-1, 2, 8) # {-1, 0, 1}
    organism[8:] = np.random.randint(-8, 9, 5) # [-8, 8] integers
    # The death lower threshold has to be lower than the death upper threshold.
    while organism[8] > organism[9]: organism[8:10] = np.random.randint(-8, 9, 2)
    # The life lower threshold has to be lower than the life upper threshld.
    while organism[10] > organism[11]: organism[10:12] = np.random.randint(-8, 9, 2)
    # Need at least 1 neighbor to spawn.
    while abs(organism[-1]) <= 0.5: organism[-1] = np.random.randint(-8, 9)
    return organism 


def mutate_real_organism(organism, incremental=False):
    mutation = random_real_organism()
    idx = np.random.randint(NUM_FEATURES)
    if incremental:
        mutation = -0.1 if np.random.rand() < 0.5 else +0.1
        organism[idx] += mutation
    else:
        organism[idx] = mutation[idx]


def mutate_integer_organism(organism, incremental=False):
    idx = np.random.randint(NUM_FEATURES)
    if incremental:
        mutation = -1 if np.random.rand() < 0.5 else +1
        organism[idx] += mutation
    else:
        mutation = random_integer_organism()
        organism[idx] = mutation[idx]
        

def mutate(organism, incremental=False):
    if USE_INTEGERS_ORGANISMS:
        mutate_integer_organism(organism, incremental)
    else:
        mutate_real_organism(organism, incremental)    
    organism[:8] = ((organism[:8] + 1) % 3) - 1 # [-1, 1] -> [0, 2], mod 3 -> [-1, 1]
    organism[8:] = organism[8:] % 9 
    
    # Enforce a "hole" around 0 as the life-from-death parameter -- need at least 1 neighbor to spawn.
    if organism[-1] == 0:
        organism[-1] = 1 if np.random.rand() < 0.5 else -1
    elif organism[-1] <= 0.5 and organism[-1] > 0: 
        organism[-1] = -0.51 
    elif organism[-1] >= -0.5 and organism[-1] < 0: 
        organism[-1] = 0.51


def random_organism():
    if USE_INTEGERS_ORGANISMS:
        return random_integer_organism()
    else:
        return random_real_organism()    


def get_new_generation_roulette_wheel(population, scores):
    score_distribution = softmax(scores)
    idxs = np.random.choice(range(len(population)), POPULATION_SIZE, p=score_distribution, replace=True)
    new_population = [population[i].copy() for i in idxs]
    for organism in new_population:
        if np.random.rand() < MUTATION_RATE:
            mutate(organism, incremental=USE_INCREMENTAL_MUTATIONS)  
    return new_population  


def top_k_ranked_organism_idxs(scores, k):
    sorted_score_idxs = sorted(list(range(len(scores))), key=lambda i: scores[i])
    return sorted_score_idxs[-k:]


def get_new_generation_truncation(population, scores, fill_with_random=False):
    survived_idxs = top_k_ranked_organism_idxs(scores, -int(len(scores)*SURVIVAL_FACTOR))
    new_population = [population[i].copy() for i in survived_idxs]
    i = 0
    while len(new_population) < POPULATION_SIZE:
        if fill_with_random:
            new_population.append( random_organism() )
        else:
            new_organism = population[survived_idxs[i]].copy()
            if np.random.rand() < MUTATION_RATE:
                mutate(new_organism, incremental=USE_INCREMENTAL_MUTATIONS)  
            new_population.append( new_organism )
            i = (i+1) % len(survived_idxs)
    return new_population


if __name__ == "__main__":
    experiment_name = f"{'roulette' if USE_ROULETTE_SELECTION else 'truncation'}{'_annealing' if USE_ANNEALING else ''}{'_incremental' if USE_INCREMENTAL_MUTATIONS else ''}_{'int' if USE_INTEGERS_ORGANISMS else 'real'}_r={MUTATION_RATE}_n={POPULATION_SIZE}" 
    time = datetime.now()
    time_string = f"{time.date()}@{time.hour}:{time.minute}:{time.second}"
    writer = SummaryWriter(f"./runs/{score_fn_desc}_{experiment_name}_{time_string}")
    population = [random_organism() for _ in range(POPULATION_SIZE)]
    for iter in range(NUM_ITERS):
        seed = centerSeedInitial(7, 7) # randomInitial(0.5) #
        x0 = np.repeat(seed[None, ...], POPULATION_SIZE, 0)
        kernel, update_rule = make_kernels(population), make_rules(population)
        final = runCellularAutomata(
            x0, 
            kernel, 
            update_rule, 
            NUM_SIM_STEPS_PER_ITER, 
            1, 
            iter, 
            display=True if (iter % DISPLAY_GENERATIONS == 0) or (iter == NUM_ITERS-1) else False, 
            plot_idxs=None if iter == 0 else top_k_ranked_organism_idxs(fitness_scores, 9),
            fullscreen_display=(iter == NUM_ITERS-1)
        )
        fitness_scores = score(final)
        if USE_ROULETTE_SELECTION:
            population = get_new_generation_roulette_wheel(population, fitness_scores)
        else:
            population = get_new_generation_truncation(population, fitness_scores, fill_with_random=False)

        if USE_ANNEALING:
            if iter == 0:
                r_m_0 = MUTATION_RATE
            if iter == NUM_ITERS // 2:
                USE_INCREMENTAL_MUTATIONS = True
                USE_ROULETTE_SELECTION = False
            MUTATION_RATE = r_m_0 * (1 - iter / NUM_ITERS)
        if iter % DISPLAY_GENERATIONS == 0:
            print(f"Average fitness @ iteration {iter}: {fitness_scores.mean()}")
        writer.add_scalar('Average Fitness Score', fitness_scores.mean(), iter)
        writer.add_scalar('Mutation Rate', MUTATION_RATE, iter)


    with open("population.pkl", "wb") as f:
        pickle.dump(population, f)
