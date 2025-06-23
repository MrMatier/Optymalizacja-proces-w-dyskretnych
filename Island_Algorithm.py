import math
import itertools
import time
import random
from os.path import isabs

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator, FormatStrFormatter

#generator liczb pseudolosowych
class RandomNumberGenerator:
    def __init__(self, seedValue=None):
        #seed
        self.__seed = seedValue or 1

    #calkowita, algorytm LCG
    def nextInt(self, low, high):
        m = 2147483647
        a = 16807
        b = 127773
        c = 2836
        k = int(self.__seed / b)
        self.__seed = a * (self.__seed % b) - k * c
        if self.__seed < 0:
            self.__seed += m
        value_0_1 = self.__seed / m
        return low + int(math.floor(value_0_1 * (high - low + 1)))

    #losowy float
    def nextFloat(self, low, high):
        val = self.nextInt(int(low*100000), int(high*100000)) / 100000.0
        return val

#generowanie instancji problemu FP||Cmax
def generate_flowshop_instance(n_jobs, n_machines, rng):

    #macierz p[j][i], j czas wywkonania zadnia na maszynie i
    p = [[rng.nextInt(1, 20) for _ in range(n_machines)] for _ in range(n_jobs)]
    return p

#f celu oblicza Cmax dla danej permutacji zadan
def compute_Cmax(permutation, p):
    m = len(p[0])                     #liczba maszyn
    C = [[0]*len(permutation) for _ in range(m)]
    for idx, job in enumerate(permutation):
        for machine in range(m):
            prev_job_end   = C[machine][idx-1] if idx>0 else 0
            prev_stage_end = C[machine-1][idx] if machine>0 else 0
            start = max(prev_job_end, prev_stage_end)
            C[machine][idx] = start + p[job][machine]
    return C[-1][-1]  #czas zakonczenia ostatniego joba na ostatniej maszynie


#brute force dla malych instancji(weryfikuje poprawnosc)
def brute_force(p):
    n_jobs = len(p)
    best_perm = None
    best_val = float('inf')
    for perm in itertools.permutations(range(n_jobs)):
        val = compute_Cmax(perm, p)
        if val < best_val:
            best_val, best_perm = val, perm
    return list(best_perm), best_val

#operatory genetyczne, losuje dwa indeksy a,b kopiuje fragment parent1[a:b] do dziecka reszta wypełniona genami z parent2 w kolejności
def order_crossover(parent1, parent2, rng):
    size = len(parent1)
    a = rng.nextInt(0, size-1)
    b = rng.nextInt(0, size-1)
    if a > b:
        a, b = b, a
    child = [-1]*size
    child[a:b+1] = parent1[a:b+1]
    ptr = 0
    for g in parent2:
        if g not in child:
            while child[ptr] != -1:
                ptr += 1
            child[ptr] = g
    return child

#mutacja wybiera dwa losowe indeksy i zamienia zadnaia
def mutate_swap(seq, rng):
    i = rng.nextInt(0, len(seq)-1)
    j = rng.nextInt(0, len(seq)-1)
    seq[i], seq[j] = seq[j], seq[i]


#algorytm wyspowy losowe permutacje przez Fisher Yates z rng, dla każdej wyspy sortujemy populację po Cmax zachowujemy 2 elity, tworzymy resztę dzieci przez crossover i mutację, aktualizujemy globalne Cmax, co migrate_interval zbieramy najlepsze migrate_size z każdej wyspy i przenosimy je między wyspami
#n_islands - liczba wysp (podpopulacji)
#pop_size - liczba permutacji na wyspe
#gens - liczba pokolen (generaci) ewolucji
#migrate_interval - co ile generacji migracja
#migrate_size - ilu najlepszych przenosic

def island_alg(p, rng, n_islands=3, pop_size=10, gens=30, migrate_interval=10, migrate_size=1):
    n_jobs = len(p)
    #inicjalizacja wysp
    islands = []
    for l in range(n_islands):
        pop = []
        for k in range(pop_size):
            perm = list(range(n_jobs))
            #Fisher Yates z rng
            for i in range(n_jobs-1, 0, -1):
                j = rng.nextInt(0, i)
                perm[i], perm[j] = perm[j], perm[i]
            pop.append(perm)
        islands.append(pop)
    best_global, best_val = None, float('inf')

    #ewolucja
    for g in range(gens):
        for isl in range(n_islands):
            pop = islands[isl]
            #selekcja i elity
            pop.sort(key=lambda perm: compute_Cmax(perm, p))
            new_pop = pop[:2]
            #tworzenie kolejnych dzieci
            while len(new_pop) < pop_size:
                i1 = rng.nextInt(0, min(4, len(pop)-1))
                i2 = rng.nextInt(0, min(4, len(pop)-1))
                child = order_crossover(pop[i1], pop[i2], rng)
                if rng.nextFloat(0,1) < 0.2:
                    mutate_swap(child, rng)
                new_pop.append(child)
            islands[isl] = new_pop
            #aktualizacja globalnego najlepszego
            val = compute_Cmax(new_pop[0], p)
            if val < best_val:
                best_val, best_global = val, new_pop[0]
        #migracja
        if (g+1) % migrate_interval == 0:
            migrants = [pop[:migrate_size] for pop in islands]
            for i in range(n_islands):
                target = (i+1) % n_islands
                islands[target][-migrate_size:] = migrants[i]
    return best_global, best_val

#test generuje instancję przez RNG, oblicza brute force optimum potem uruchamia island i porównuje wyniki
def test_island(num_tests=10):
    successes = 0
    for seed in range(num_tests):
        rng = RandomNumberGenerator(seedValue=seed)
        p = generate_flowshop_instance(n_jobs=6, n_machines=3, rng=rng)
        rng_is = RandomNumberGenerator(seedValue=seed)
        bf_perm, bf_val = brute_force(p)
        is_perm, is_val = island_alg(p, rng_is)
        print(f"test {seed}: brute={bf_val}, island={is_val}")
        if is_val == bf_val:
            successes += 1
    print(f"sukcesy: {successes}/{num_tests}")

#konwergencja dla n_jobs=20
def island_convergence(p, rng, n_islands=3, pop_size=10, gens=15, migrate_interval=5, migrate_size=1):
    islands = []
    n_jobs = len(p)
    for l in range(n_islands):
        pop = [list(range(n_jobs)) for k in range(pop_size)]
        for perm in pop:
            random.shuffle(perm)
        islands.append(pop)

    best_val = float('inf')
    history = []
    for g in range(gens):
        for isl in range(n_islands):
            pop = islands[isl]
            pop.sort(key=lambda x: compute_Cmax(x, p))
            new_pop = pop[:2]
            while len(new_pop) < pop_size:
                i1 = rng.nextInt(0, min(4, len(pop)-1))
                i2 = rng.nextInt(0, min(4, len(pop)-1))
                child = order_crossover(pop[i1], pop[i2], rng)
                if rng.nextFloat(0,1) < 0.2:
                    mutate_swap(child, rng)
                new_pop.append(child)
            islands[isl] = new_pop
        if (g+1) % migrate_interval == 0:
            migrants = [pop[:migrate_size] for pop in islands]
            for i in range(n_islands):
                islands[(i+1) % n_islands][-migrate_size:] = migrants[i]
        for pop in islands:
            cand = compute_Cmax(pop[0], p)
            if cand < best_val:
                best_val = cand
        history.append(best_val)
    return history

#skalowalność bruteforce vs wyspowy w log
def experiment_scalability(jobs_range=range(1,11), seeds=3, n_islands=3):
    print("n_jobs | brute[s]  |  island[s]")
    brute_times, is_times = [], []
    for n in jobs_range:
        tb = tg = 0.0
        for seed in range(seeds):
            rng = RandomNumberGenerator(seed)
            p = generate_flowshop_instance(n, 3, rng)
            t0 = time.perf_counter(); brute_force(p); tb += time.perf_counter() - t0
            t0 = time.perf_counter(); island_alg(p, RandomNumberGenerator(seed), n_islands=n_islands); tg += time.perf_counter() - t0
        avg_b, avg_g = tb/seeds, tg/seeds
        brute_times.append(avg_b); is_times.append(avg_g)
        print(f"  {n:2d}    | {avg_b:7.4f} | {avg_g:7.4f}")

    plt.figure(figsize=(7,4))
    plt.plot(list(jobs_range), brute_times, marker='o', label='brute force')
    plt.plot(list(jobs_range), is_times, marker='o', label='island')
    plt.yscale('log')
    plt.xlabel('liczba zadań')
    plt.ylabel('czas[s]')
    plt.title('skalowalność brute force vs island')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.minorticks_off()
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.4f}"))
    plt.tight_layout()
    plt.show()

#stala liczba zadań n_jobs=6 a rózna ilosc wysp
def experiment_islands(sizes=(1,5,10), seeds=5):
    print("\nn_islands | srednie ratio do optimum")
    for ni in sizes:
        ratios = []
        for seed in range(seeds):
            rng1 = RandomNumberGenerator(seed)
            p6 = generate_flowshop_instance(6, 3, rng1)
            _, bf = brute_force(p6)
            _, isla = island_alg(p6, RandomNumberGenerator(seed), n_islands=ni)
            ratios.append(isla/bf)
        print(f"   {ni:2d}     | {sum(ratios)/len(ratios):.3f}")

#rosnąca liczba zadań n_jobs a rózna ilosc wysp
def experiment_islands2(jobs_range=range(1, 11), n_machines=3, n_islands=5, seeds=3):
    times = []
    print(f"\nisland, liczba wysp = {n_islands}")
    print("n_jobs | island [s]")
    for n_jobs in jobs_range:
        is_times = []
        for seed in range(seeds):
            rng = RandomNumberGenerator(seed)
            p = generate_flowshop_instance(n_jobs, n_machines, rng)
            t0 = time.perf_counter()
            island_alg(p, RandomNumberGenerator(seed),
                       n_islands=n_islands, pop_size=10, gens=30)
            is_times.append(time.perf_counter() - t0)
        avg = sum(is_times)/seeds
        times.append((n_jobs, avg))
        print(f" {n_jobs:6d} | {avg:13.4f}")
    return times

def plot_islands2(times):
    jobs = [r[0] for r in times]
    isla = [r[1] for r in times]
    plt.figure(figsize=(7,4))
    plt.plot(jobs, isla, marker='o', label='island')
    plt.xlabel('liczba zadań')
    plt.ylabel('średni czas[s]')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.minorticks_off()
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_island(10)

    #krzywa konwergencji
    rng = RandomNumberGenerator(42)
    p20 = generate_flowshop_instance(20, 3, rng)
    hist = island_convergence(p20, RandomNumberGenerator(42))
    print("konwergencja (n_jobs=20):", hist)
    plt.figure(figsize=(7,4))
    plt.plot(range(1,len(hist)+1), hist, marker='o')
    plt.xlabel('pokolenie'); plt.ylabel('najlepsze Cmax')
    plt.title('konwergencja dla 20 zadań ')
    plt.grid(True)
    plt.minorticks_off()
    plt.tight_layout()
    plt.show()

    #skalowalność
    experiment_scalability()

    #stala liczba zadań n_jobs=6 a rózna ilosc wysp
    experiment_islands()

    #rosnąca liczba zadań n_jobs a rózna ilosc wysp
    for ni in (5, 10, 15):
        times = experiment_islands2(range(1, 11), n_machines=3, n_islands=ni)
        plot_islands2(times)
