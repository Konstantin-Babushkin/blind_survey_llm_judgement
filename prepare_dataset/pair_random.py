import random, itertools

tasks = ['C13','C14','C15','C16','C17','C18']
pairs = list(itertools.combinations(tasks, 2))
random.shuffle(pairs)
assignment = pairs[:5]
for i, pair in enumerate(assignment, 1):
    print(f'{pair[0]} + {pair[1]}')
