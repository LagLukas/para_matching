import os
import json

SEED = 10
MAX_THREADS = 12
folder = "../results"

def execute_cpp_experiment(seed, size, dim, iterations, match_perc, n_threads):
    command = "../build/main {} {} {} {} {} {}".format(seed, size, dim, iterations, match_perc, n_threads)
    res = os.popen(command).read()
    result_dict = {}
    result_dict["seq"] = []
    result_dict["mutex"] = []
    result_dict["local_list"] = []
    lines = res.split("\n")
    for line in lines:
        entries = line.split()
        if len(entries) == 3:
            result_dict["seq"].append(float(entries[0]))
            result_dict["mutex"].append(float(entries[1]))
            result_dict["local_list"].append(float(entries[2]))
    return result_dict

def execute_experiment(seed, size, dim, iterations, match_perc, n_threads):
    res = execute_cpp_experiment(seed, size, dim, iterations, match_perc, n_threads)
    match_conv = str(match_perc).replace(".", "x")
    file_path = folder + os.sep + "{}_{}_{}_{}_{}_{}.json".format(seed, size, dim, iterations, match_perc, n_threads)
    with open(file_path, "w") as file:
        json.dump(res, file, sort_keys=True, indent=4)

def execute_experiment_series(seed, iterations, sizes, dims, match_percs, threads):
    n_exp = len(sizes) * len(dims) * len(match_percs) * len(threads)
    counter = 0
    for size in sizes:
        for dim in dims:
            for match_perc in match_percs:
                for n_threads in threads:
                    counter += 1
                    execute_experiment(seed, size, dim, iterations, match_perc, n_threads)
                    print("finished " + str(counter / n_exp) + "percent of all experiments") 

if __name__== "__main__":
    seed = SEED
    iterations = 100
    sizes = list(map(lambda x: 10000 * x, range(1, 16)))
    dims = list(map(lambda x: 3 * x, range(1, 9)))
    match_percs = list(map(lambda x: 0.01 * x, range(1, 31)))
    threads = list(map(lambda x: 2 * x, range(1, int(MAX_THREADS / 2) + 1)))
    execute_experiment_series(seed, iterations, sizes, dims, match_percs, threads)
