import numpy as np
import multiprocessing
import json
import sys
import matplotlib.pyplot as plt
import tikzplotlib as tkz
import datetime

from running_utils import trial

if __name__ == '__main__':

    with open(sys.argv[1]) as json_file:
        config = json.load(json_file)

    time_print = datetime.datetime.now().strftime("(%Y-%b-%d %I:%M%p)")

    S = config["S"]
    A = config["A"]
    H = config["H"]
    K = config["K"]
    alg_lst = config["alg_lst"]
    n_trials = config["n_trials"]
    n_cores = config["n_cores"]

    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.starmap(trial, [(S, A, H, K, alg_lst, i) for i in range(n_trials)])

    results_dict = {alg : [] for alg in alg_lst}
    results_dict["config"] = config

    for i in range(len(results)):
        for alg in alg_lst:
            results_dict[alg].append(list(results[i][alg]))

    with open(f"results/rawdata_{str(config)}_time={time_print}.json", "w") as fp:
        json.dump(results_dict, fp, indent=4)

    plt.figure()

    for alg in alg_lst:

        x_plt = np.linspace(0, K-1, 100, dtype=int)
        results_alg = np.array(results_dict[alg]).T
        results_mean = results_alg.mean(axis=1)
        results_std = 1.96 * results_alg.std(axis=1) / np.sqrt(n_trials)
        plt.plot(x_plt, results_mean[x_plt], label=alg)
        plt.fill_between(x_plt, results_mean[x_plt] - results_std[x_plt],
                         results_mean[x_plt] + results_std[x_plt], alpha=0.3)

    plt.legend()
    plt.savefig(f"results/figplot_{str(config)}_time={time_print}.jpg")
    tkz.save(f"results/figplot_{str(config)}_time={time_print}.tex")
    plt.yscale("log")
    plt.savefig(f"results/figplot_{str(config)}_log_time={time_print}.jpg")
