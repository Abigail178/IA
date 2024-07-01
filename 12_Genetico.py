#=========================================
# Algoritmo genetico simple
#=========================================
import datetime
import random

random.seed(random.random())
startTime = datetime.now()

#===========================
# los genes
#===========================
geneSet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

#======================
# Objetivo
#======================
target = 'Hola Mundo'

#======================
# Frase inicial
#======================
def generate_parent(length):
    genes = []
    while len(genes) < length:
        sample_size = min(length - len(genes), len(geneSet))
        genes.extend(random.sample(geneSet, sample_size))
    return ''.join(genes)

#============================
# Funcion de aptitud
#============================
def get_fitness(guess):
    return sum(1 for expected, actual in zip(target, guess) if expected == actual)


#================================
# Mutacion de letras en la frase
#================================
def mutate(parent):
    index = random.randrange(0, len(parent))
    child_genes = list(parent)
    new_gene, alternate = random.sample(geneSet, 2)
    child_genes[index] = alternate if new_gene == child_genes[index] else new_gene
    return ''.join(child_genes)


#=============================
# Monitoreo de la solucion
#=============================
def display(guess):
    time_diff = datetime.datetime.now() - startTime
    fitness = get_fitness(guess)
    print("{}\t{}\t{}".format(guess, fitness, time_diff))

#==============================
# Codigo principal
#==============================
best_parent = generate_parent(len(target))
best_fitness = get_fitness(best_parent)
display(best_parent)

#====================
# Iteraciones
#====================
while True:
    child = mutate(best_parent)
    child_fitness = get_fitness(child)
    if best_fitness >= child_fitness:
        display(child)
        continue
    display(child)
    if child_fitness >= len(best_parent):
        break
    best_fitness = child_fitness
    best_parent = child