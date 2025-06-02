import numpy as np
import multiprocessing
import json
import sys
import matplotlib.pyplot as plt
import tikzplotlib as tkz
import datetime, time

class BernoulliBanditEnv:

    def __init__(self, A):
        self.A = A
        self.exp_vals = np.random.rand(A)

    def step(self, action):
        return np.random.uniform(0,1,1) < self.exp_vals[action]

    def get_optimal_exp_value(self):
        return max(self.exp_vals)


class UCBeff:
    
    def __init__(self, A, exp_param=0.5):
        self.A = A
        self.t = 1
        self.exp_param = exp_param
        self.last_pull = None
        self.avg_reward = np.zeros(self.A)
        self.n_pulls = np.zeros(self.A, dtype=int)

    def choose(self):
        ucb1 = [self.avg_reward[a] + np.sqrt(self.exp_param * np.log(self.t) / self.n_pulls[a]) for a in range(self.A)]
        self.last_pull = np.argmax(ucb1)
        return self.last_pull

    def update(self, reward):
        self.t += 1
        self.avg_reward[self.last_pull] = (self.avg_reward[self.last_pull] * self.n_pulls[
                                           self.last_pull] + reward) / (self.n_pulls[self.last_pull] + 1)
        self.n_pulls[self.last_pull] += 1

class UCB:
    
    def __init__(self, A, T):
        self.A = A
        self.T = T
        self.t = 1
        self.last_pull = None
        self.avg_reward = np.zeros(self.A)
        self.n_pulls = np.zeros(self.A, dtype=int)

    def choose(self):
        ucb1 = [self.avg_reward[a] + np.sqrt(2 * np.log(self.T) / self.n_pulls[a]) for a in range(self.A)]
        self.last_pull = np.argmax(ucb1)
        return self.last_pull

    def update(self, reward):
        self.t += 1
        self.avg_reward[self.last_pull] = (self.avg_reward[self.last_pull] * self.n_pulls[
                                           self.last_pull] + reward) / (self.n_pulls[self.last_pull] + 1)
        self.n_pulls[self.last_pull] += 1


def trial(A, T, run_lst, trial_id):
    np.random.seed(trial_id)
    env = BernoulliBanditEnv(A)              
    opt = env.get_optimal_exp_value()
    opt_rewards = opt * np.ones(T)
    results_all = {}
    for alg in run_lst:
        start_time = time.time()
        time_print = datetime.datetime.now().strftime("(%Y-%b-%d %I:%M%p)")
        print(time_print + " Starting trial " + str(trial_id+1) + " for " + alg + ".")
        if alg == "UCB":
            agent = UCB(A, T)
        elif alg == "UCBeff":
            agent = UCBeff(A)
        else:
            raise ValueError("Error in input")
        rewards = np.zeros(T)
        for t in range(T):
            action = agent.choose()
            rewards[t] = env.step(action)
            agent.update(rewards[t])
        ep_regret = opt_rewards - rewards
        ep_regret = ep_regret.cumsum()
        time_print_end = datetime.datetime.now().strftime("(%Y-%b-%d %I:%M%p)")
        print(time_print_end + " Terminating trial " + str(trial_id+1) + " for " + alg + ". Elapsed time: " + str(int(time.time() - start_time)) + " sec.")
        results_all[alg] = ep_regret
    return results_all

if __name__ == '__main__':

    A = int(sys.argv[1])
    T = int(sys.argv[2])
    n_trials = int(sys.argv[3])
    n_cores = int(sys.argv[4])
    
    # A = 3
    # T = 100000
    # n_trials = 20
    # n_cores = 4
    alg_lst = ["UCB", "UCBeff"]
    
    with multiprocessing.Pool(processes=n_cores) as pool:
        results = pool.starmap(trial, [(A, T, alg_lst, i) for i in range(n_trials)])
    
    # results = [trial(A, T, alg_lst, i) for i in range(n_trials)]
    
    results_dict = {alg : [] for alg in alg_lst}
    results_dict["config"] = {"A": A, "T": T, "n_trials": n_trials, "alg_lst": alg_lst}
    
    for i in range(len(results)):
        for alg in alg_lst:
            results_dict[alg].append(list(results[i][alg]))
    
    time_print = datetime.datetime.now().strftime("(%Y-%b-%d %I-%M%p)")
    with open(f"results/bandits_rawdata_A{str(A)}_T{str(T)}_time={time_print}.json", "w") as fp:
        json.dump(results_dict, fp, indent=4)
    
    plt.figure()
    for alg in alg_lst:
        x_plt = np.linspace(0, T-1, 100, dtype=int)
        results_alg = np.array(results_dict[alg]).T
        results_mean = results_alg.mean(axis=1)
        results_std = 1.96 * results_alg.std(axis=1) / np.sqrt(n_trials)
        plt.plot(x_plt, results_mean[x_plt], label=alg)
        plt.fill_between(x_plt, results_mean[x_plt] - results_std[x_plt],
                         results_mean[x_plt] + results_std[x_plt], alpha=0.3)
    plt.legend()
    plt.savefig(f"results/bandits_figplot_A{A}_T{T}_time={time_print}.jpg")
    tkz.save(f"results/bandits_figplot_A{A}_T{T}_time={time_print}.tex")