import numpy as np



class UCBVI:


    def __init__(self, S, A, H, K, R, bound, improved=False):

        assert isinstance(S, int) and isinstance(A, int) and isinstance(H, int) and isinstance(K, int), "Error in S, A, H or K: they must be all int"
        assert S > 0 and A > 0 and H > 0 and K > 0, "Error in S, A, H or K: they must be all positive"

        assert isinstance(R, np.ndarray), "Error in R: it must be instance of np.ndarray"
        assert R.ndim == 2, "Error in R.ndim"
        assert R.shape == (S, A), "Error R shape"

        self.S = S
        self.A = A
        self.H = H
        self.K = K
        self.R = R

        self.L = np.log(5 * self.S * self.A * self.K * self.H)

        self.Nxay = np.zeros((self.S, self.A, self.S))
        self.Nxa = np.zeros((self.S, self.A))
        self.Phat = np.zeros((self.S, self.A, self.S))

        assert bound == "CH" or bound == "BF", "Error in the selected bound: it must be CH or BF"

        self.empirical = 0 if bound == "CH" else 1

        self.improved = improved

        if self.empirical == 0:
            if self.improved:
                self.Bxa = 7 * self.H * self.L * np.ones((self.S, self.A))
            else:
                self.Bxa = 2 * self.H * self.L * np.ones((self.S, self.A))
        else:
            self.Nxah = np.zeros((self.S, self.A, self.H+1))
            if self.improved:
                self.emp_cost = 7056 * pow(self.H, 3) * pow(self.S, 2) * self.A * pow(self.L, 2)
            else:
                self.emp_cost = 10000 * pow(self.H, 3) * pow(self.S, 2) * self.A * pow(self.L, 2)

        self.oldQstar = None
        
        self.newEpisode()



    def newEpisode(self):

        self.state = None
        self.lastaction = None

        self.Qstar = np.zeros((self.S, self.A, self.H))
        self.Vstar = np.zeros((self.S, self.H+1))

        for h in range(self.H-1, -1, -1):

            for s in range(self.S):

                for a in range(self.A):

                    if self.Nxa[s, a] > 0:

                        if self.empirical == 0:
                            bound = self.Bxa[s, a]
                        else:
                            aux = 0
                            for s_prime in range(self.S):
                                aux = aux + self.Phat[s, a, s_prime] * min(
                                    pow(self.H, 2), self.emp_cost / max(1, np.sum(self.Nxah[s_prime, : , h+1]))
                                )
                            if self.improved:
                                bound = (7 / 3) * self.H * self.L / max(self.Nxa[s, a] - 1, 1) + np.sqrt(4 * self.L * pow(
                                    np.std(self.Phat[s, a, :] * self.Vstar[:, h+1]), 2) / self.Nxa[s, a]
                                    ) + np.sqrt(4 * np.sum(aux) / self.Nxa[s, a])
                            else:
                                bound = (14 / 3) * self.H * self.L / self.Nxa[s, a] + np.sqrt(8 * self.L * pow(
                                    np.std(self.Phat[s, a, :] * self.Vstar[:, h+1]), 2) / self.Nxa[s, a]
                                    ) + np.sqrt(8 * np.sum(aux) / self.Nxa[s, a])

                        self.Qstar[s, a, h] = self.R[s, a] + self.Phat[s, a, :].reshape(
                                1, self.S) @ self.Vstar[:, h+1].reshape(self.S, 1) + bound

                        if self.Qstar[s, a, h] > self.H:
                            self.Qstar[s, a, h] = self.H

                        if self.oldQstar is not None:
                            self.Qstar[s, a, h] = min(self.Qstar[s, a, h], self.oldQstar[s, a, h])

                    else:

                        self.Qstar[s, a, h] = self.H

                self.Vstar[s, h] = max(self.Qstar[s, :, h])

        self.oldQstar = self.Qstar



    def choose(self, state, stage):

        self.state = state
        self.stage = stage

        self.lastaction = np.argmax(self.Qstar[self.state, :, stage])

        if isinstance(self.lastaction, np.ndarray):
            self.lastaction = np.random.choice(self.lastaction)

        return self.lastaction



    def update(self, newstate, reward): # reward is given, so this input is ignored

        assert isinstance(newstate, int) or isinstance(newstate, np.int64), "Error in update(): newstate must be an integer"
        assert newstate >= 0 and newstate < self.S, "Error in update(): newstate must be s.t.: 0<= state < self.S"

        if self.state is not None and self.lastaction is not None:

            self.Nxay[self.state, self.lastaction, newstate] = self.Nxay[self.state, self.lastaction, newstate] + 1
            self.Nxa[self.state, self.lastaction] = self.Nxa[self.state, self.lastaction] + 1

            self.Phat[self.state, self.lastaction, :] = \
                self.Nxay[self.state, self.lastaction, :] / self.Nxa[self.state, self.lastaction]

            if self.empirical == 0:
                if self.improved:
                    self.Bxa[self.state, self.lastaction] = 7 * self.H * self.L * np.sqrt(1 / self.Nxa[self.state, self.lastaction])
                else:
                    self.Bxa[self.state, self.lastaction] = 2 * self.H * self.L * np.sqrt(1 / self.Nxa[self.state, self.lastaction])

            else:
                self.Nxah[self.state, self.lastaction, self.stage] = self.Nxah[self.state, self.lastaction, self.stage] + 1

        self.state = newstate




class MVP:


    def __init__(self, S, A, H, K, R, delta):

        assert (isinstance(S, int) and isinstance(A, int) and isinstance(H, int)
                and isinstance(K, int)), "Error in S, A, H or K: they must be all int"
        assert S > 0 and A > 0 and H > 0 and K > 0, "Error in S, A, H or K: they must be all positive"

        assert isinstance(R, np.ndarray), "Error in R: it must be instance of np.ndarray"
        assert R.ndim == 2, "Error in R.ndim"
        assert R.shape == (S, A), "Error R shape"

        assert isinstance(delta, float), "Error in delta"
        assert 1 > delta > 0, "delta must be s.t.: 0 < delta < 1"

        self.S = S
        self.A = A
        self.H = H
        self.K = K
        self.R = R
        self.delta = delta

        self.c1 = 460 / 9
        self.c3 = 544 / 9
        self.logrecdeltaprime = np.log(1 / (self.delta / (200 * self.S * self.A * pow(self.H, 2) * pow(self.K, 2))))

        self.Nxa = np.zeros((self.S, self.A))
        self.Nxay = np.zeros((self.S, self.A, self.S))
        self.NxaALL = np.zeros((self.S, self.A))
        self.Phat = np.zeros((self.S, self.A, self.S))

        self.Qstar = self.H * np.ones((self.S, self.A, self.H))
        self.Vstar = self.H * np.ones((self.S, self.H+1))

        self.update_needed = False

        self.epoch_lst = [1]
        power = 1
        while power < self.H * self.K:
            power *= 2
            self.epoch_lst.append(power)

        self.newEpisode()



    def newEpisode(self):

        """
        if self.ep_count % 100000 == 0:
            print("Qstar")
            print(self.Qstar)
        """

        self.state = None
        self.lastaction = None

        if self.update_needed:

            self.Qstar = np.zeros((self.S, self.A, self.H))
            self.Vstar = np.zeros((self.S, self.H+1))

            for h in range(self.H-1, -1, -1):

                for s in range(self.S):

                    for a in range(self.A):

                        bound = self.c1 * np.sqrt(
                            pow(np.std(self.Phat[s, a, :] * self.Vstar[:, h+1]), 2) * self.logrecdeltaprime / max(1, self.Nxa[s, a])
                        ) + self.c3 * self.H * self.logrecdeltaprime / max(1, self.Nxa[s, a])

                        self.Qstar[s, a, h] = self.R[s, a] + self.Phat[s, a, :].reshape(
                                1, self.S) @ self.Vstar[:, h+1].reshape(self.S, 1) + bound

                        self.Qstar[s, a, h] = min(self.Qstar[s, a, h], self.H)

                    self.Vstar[s, h] = max(self.Qstar[s, :, h])

            self.update_needed = False



    def choose(self, state, stage):

        self.state = state
        self.stage = stage

        self.lastaction = np.argmax(self.Qstar[self.state, :, stage])

        if isinstance(self.lastaction, np.ndarray):
            self.lastaction = np.random.choice(self.lastaction)

        return self.lastaction



    def update(self, newstate, reward): # reward is given, so this input is ignored

        assert isinstance(newstate, int) or isinstance(newstate, np.int64), "Error in update(): newstate must be an integer"
        assert newstate >= 0 and newstate < self.S, "Error in update(): newstate must be s.t.: 0 <= state < self.S"

        if self.state is not None and self.lastaction is not None:

            self.Nxay[self.state, self.lastaction, newstate] = self.Nxay[self.state, self.lastaction, newstate] + 1
            self.NxaALL[self.state, self.lastaction] = self.NxaALL[self.state, self.lastaction] + 1

            if self.NxaALL[self.state, self.lastaction] in self.epoch_lst:

                self.Nxa[self.state, self.lastaction] = np.sum(self.Nxay[self.state, self.lastaction, :])
                self.Phat[self.state, self.lastaction, :] = self.Nxay[self.state, self.lastaction, :] / self.Nxa[self.state, self.lastaction]
                self.Nxay[self.state, self.lastaction, :] = 0
                self.update_needed = True
