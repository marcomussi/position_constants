import numpy as np

class ToyEnv:

    # K (number of episodes) not needed right now
    # H (number of stages) needed in order to run value iteration
    # the P and R matrices can be given or not
    # the noise is not seeded
    # P and R are not stage-dependent

    
    def __init__(self, S, A, H, P=None, R=None, initS=0): 
        
        assert isinstance(S, int) and isinstance(A, int) and isinstance(H, int), "Error in S, A or H: they must be all int"
        assert S > 0 and A > 0 and H > 0, "Error in S, A or H: they must be all positive"
        
        self.S = S
        self.A = A
        self.H = H
        
        # P has dimension (S, A, S)
        # position 0: current state, position 1: taken action, position 2: next state
        if P is not None:
            assert isinstance(P, np.ndarray), "Error in P: it must be instance of np.ndarray"
            assert P.ndim == 3, "Error in P.ndim"
            assert P.shape == (self.S, self.A, self.S), "Error P shape"
            self.P = P
        else:
            self.P = np.random.uniform(low=0.0, high=1.0, size=(self.S, self.A, self.S))
            for s in range(self.S):
                for a in range(self.A):
                    self.P[s, a] = self.P[s, a] / np.sum(self.P[s, a])
        
        if R is not None:
            assert isinstance(R, np.ndarray), "Error in R: it must be instance of np.ndarray"
            assert R.ndim == 2, "Error in R.ndim"
            assert R.shape == (self.S, self.A), "Error R shape"
            self.R = R
        else:
            self.R = np.random.uniform(low=0.0, high=1.0, size=(self.S, self.A))
        
        self.reset(initS)
        

    
    def reset(self, initS=0):
        
        assert isinstance(initS, int) or isinstance(initS, np.int64), "Error in initS: it must be int"
        assert initS >= 0 and initS < self.S, "Initial state not consistent"
        self.currentS = initS


    
    def step(self, action):

        assert isinstance(action, int) or isinstance(action, np.int64), "Error in step(): action must be an integer"
        assert action >= 0 and action < self.A, "Error in step(): action must be s.t.: 0<= action < self.A"
        
        reward = self.R[self.currentS, action]
        
        self.currentS = np.random.choice(self.S, 1, p=self.P[self.currentS, action, :])

        if isinstance(self.currentS, np.ndarray):
            self.currentS = self.currentS[0]
        
        return self.currentS, reward


    
    def _getProbabilities(self):
        
        return self.P

    
    
    def getRewards(self):
        
        return self.R


    
    def getCurrentState(self):
        
        return self.currentS


    
    def computeOptimalValueFunction(self, initS=0):

        assert isinstance(initS, int) or isinstance(initS, np.int64), "Error in initS: it must be int"
        assert initS >= 0 and initS < self.S, "Initial state not consistent"

        Qstar = np.zeros((self.S, self.A, self.H))
        Vstar = np.zeros((self.S, self.H+1))

        for h in range(self.H-1, -1, -1):
            
            for s in range(self.S):
                
                for a in range(self.A):
            
                    Qstar[s, a, h] = self.R[s, a] + self.P[s, a, :].reshape(1, self.S) @ Vstar[:, h+1].reshape(self.S, 1)

                Vstar[s, h] = max(Qstar[s, :, h])

        return Vstar[initS, 0] # return expected value function for the initial state "initS"