import numpy as np
import multiprocessing
import json
import sys
import matplotlib.pyplot as plt
import tikzplotlib as tkz
import datetime

from running_utils import trial_rs

def generate_river(n, small=0.005, large=1):
    nA=2
    nS=n
    p = compute_probabilities(nS, nA) 
    r = compute_rewards(nS, nA, small, large)
    return p, r
         
    
def compute_probabilities(nS, nA):
    p=np.zeros((nS, nA, nS))
    for i in range(1, nS):
        p[i, 0, i-1]=1
        if i!=nS-1:
            p[i, 1, i-1]=0.1
            p[i, 1, i]=0.6
        else:
            p[i, 1, i-1]=0.7
            p[i, 1, i]=0.3
    for i in range(nS-1):
        p[i, 1, i+1]=0.3
    #state 0
    p[0, 0, 0]=1
    p[0, 1,  0]=0.7
    
    return p

def compute_rewards(nS, nA, small, large):
    r = np.zeros((nS, nA, nS))
    r[0, 0, 0]=small
    r[nS-1, 1, nS-1]=large
    return r


if __name__ == '__main__':

    time_print = datetime.datetime.now().strftime("(%Y-%b-%d %I:%M%p)")
    
    S = 10 # 5
    A = 2
    H = 25 # 10
    K = 100000 # 100000
    alg_lst = ["BF", "BFI", "MVP"]
    n_trials = 4
    n_cores = 4

    P, R = generate_river(S)
    R = np.sum(R*P, axis=-1) / 0.3

    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.starmap(trial_rs, [(S, A, H, K, P, R, alg_lst, i) for i in range(n_trials)])

    results_dict = {alg : [] for alg in alg_lst}

    for i in range(len(results)):
        for alg in alg_lst:
            results_dict[alg].append(list(results[i][alg]))

    with open(f"results/rawdata_rs_time={time_print}.json", "w") as fp:
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
    plt.savefig(f"results/figplot_rs_time={time_print}.jpg")
    tkz.save(f"results/figplot_rs_time={time_print}.tex")
    plt.yscale("log")
    plt.savefig(f"results/figplot__rs_log_time={time_print}.jpg")
