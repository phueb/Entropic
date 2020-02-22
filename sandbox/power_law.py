import random

population = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def power_cum_weights(population_size, alpha=3.0):
    """returns cum weights for power law"""
    logits = [(xi + 1) ** alpha for xi in range(population_size)]
    return [l / logits[-1] for l in logits]


cum_weights = power_cum_weights(len(population))
print(cum_weights)

sample = random.choices(population, cum_weights=cum_weights, k=1000)
for pi in population:
    print(f'{"*" * sample.count(pi)}')
