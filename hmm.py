import copy, collections
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
# \lambda = (\pi, A, B)
class HMM(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def read_params(self):
        try:
            line = raw_input()
            self.num_s, self.num_o = tuple(map(int, line.split()))
            line = raw_input()
            self.Status = Series(line.split())
            line = raw_input()
            self.Observations = Series(line.split())
            line = raw_input()
            self.pi = np.array(map(float, line.split()))
            if len(self.pi) != self.num_s:
                raise IOError('Bad params: pi')
            self.A = np.zeros((self.num_s, self.num_s))
            for i in range(self.num_s):
                line = raw_input()
                vs = map(float, line.split())
                if len(vs) != self.num_s:
                    raise IOError('Bad params: A')
                self.A[i, :] = vs
            self.B = np.zeros((self.num_s, self.num_o))
            for i in range(self.num_s):
                line = raw_input()
                vs = map(float, line.split())
                if len(vs) != self.num_o:
                    raise IOError('Bad params: B')
                self.B[i, :] = vs
            self.S2idx = Series(range(self.num_s), index=self.Status)
            self.O2idx = Series(range(self.num_o), index=self.Observations)
            return True
        except Exception, e:
            print e
            return False

    def set_params(self, Status, Observations, pi, A, B):
        self.Status = copy.copy(Status)
        self.Observations = copy.copy(Observations)
        self.num_s = len(self.Status)
        self.num_o = len(self.Observations)
        self.S2idx = Series(range(self.num_s), index=self.Status)
        self.O2idx = Series(range(self.num_o), index=self.Observations)
        self.pi = copy.copy(pi)
        self.A = copy.copy(A)
        self.B = copy.copy(B)

    def init_params(self, Status, Observations):
        '''
        for Baum_Welch_algorithm
        '''
        self.Status = copy.copy(Status)
        self.Observations = copy.copy(Observations)
        self.num_s = len(self.Status)
        self.num_o = len(self.Observations)
        self.S2idx = Series(range(self.num_s), index=self.Status)
        self.O2idx = Series(range(self.num_o), index=self.Observations)
        self.pi = np.full(self.num_s, 1.0 / self.num_s)
        self.A = np.full((self.num_s, self.num_s), 1.0 / self.num_s)
        self.B = np.full((self.num_s, self.num_o), 1.0 / self.num_o)

    def calc_alpha(self, obs):
        '''
        alpha[i, t] = P(o1, o2, ...., ot, qt = i | lambda)
        means: given model lambda, at time t, the probobility that (status is i and part of observation sequence is o1, o2, ..., ot)
        '''
        num_obs = len(obs)
        alpha = np.zeros((self.num_s, num_obs))
        oidx = self.O2idx[obs[0]]
        for i in range(self.num_s):
            alpha[i, 0] = self.pi[i] * self.B[i, oidx]
        for t in range(1, num_obs):
            oidx = self.O2idx[obs[t]]
            for j in range(self.num_s):
                tmp = 0
                for i in range(self.num_s):
                    tmp += alpha[i, t - 1] * self.A[i, j]
                alpha[j, t] = tmp * self.B[j, oidx]
        return alpha

    def calc_beta(self, obs):
        '''
        beta[i, t] = P(o_{t + 1}, o_{t + 2}, ..., o_T, q_t = i | lambda)
        means: given model lambda, at time t, the probobility that {status is i and part of observation sequence is o_{t + 1}, ..., o_T)
        '''
        num_obs = len(obs)
        beta = np.zeros((self.num_s, num_obs))
        beta[:, -1] = 1
        for t in range(num_obs - 2, -1, -1):
            oidx_t1 = self.O2idx[obs[t + 1]]
            for i in range(self.num_s):
                tmp = 0
                for j in range(self.num_s):
                    tmp += self.A[i, j] * self.B[j, oidx_t1] * beta[j, t + 1]
                beta[i, t] = tmp
        return beta

    def calc_vertibi_and_prev(self, obs):
        '''
        vertibi[i, t] = max_{q1, q2, ..., q_{t - 1}} P(q1, q2, ..., q_{t - 1}, q_t = i, o1, o2, ..., o_t | lambda)
        means: given model lambda, at time t, the max probobility that
        (status is i, part of observation sequence is o_{t + 1}, ..., o_T and the best status sequence is q1, q2, ..., q_t)
        '''
        num_obs = len(obs)
        vertibi = np.zeros((self.num_s, num_obs))
        prev = np.zeros((self.num_s, num_obs))
        oidx = self.O2idx[obs[0]]
        for i in range(self.num_s):
            vertibi[i, 0] = self.pi[i] * self.B[i, oidx]
        # prev[:, 0] = 0 # already is
        for t in range(1, num_obs):
            oidx = self.O2idx[obs[t]]
            for j in range(self.num_s):
                best_prob, best_prev = vertibi[0, t - 1] * self.A[0, j], 0
                for i in range(1, self.num_s):
                    tmp_prob = vertibi[i, t - 1] * self.A[i, j]
                    if tmp_prob > best_prob:
                        best_prob = tmp_prob
                        best_prev = i
                vertibi[j, t] = best_prob * self.B[j, oidx]
                prev[j, t] = best_prev
        return vertibi, prev

    def get_prob_of_observation_forward(self, obs):
        num_obs = len(obs)
        if num_obs == 0:
            return None
        alpha = self.calc_alpha(obs)
        if self.verbose:
            print 'alpha:\n', alpha
        return alpha[:, num_obs - 1].sum()

    def get_prob_of_observation_backward(self, obs):
        num_obs = len(obs)
        if num_obs == 0:
            return None
        beta = self.calc_beta(obs)
        if self.verbose:
            print 'beta:\n', beta
        return (self.pi * self.B[:, self.O2idx[obs[0]]] * beta[:, 0]).sum()

    def get_best_status_sequence(self, obs):
        best_status = []
        num_obs = len(obs)
        if num_obs == 0:
            return best_status
        vertibi, prev = self.calc_vertibi_and_prev(obs)
        tmp_status = np.argmax(vertibi[:, -1])
        best_status.append(tmp_status)
        for t in range(num_obs - 1, 0, -1):
            tmp_status = prev[tmp_status, t]
            best_status.append(tmp_status)
        best_status = best_status[::-1]
        if self.verbose:
            print 'vertibi:\n', vertibi
            print 'prev:\n', prev
        return self.Status[best_status]

    def Baum_Welch_algorithm(self, Status, Observations, obs, max_iter = 100):
        '''
        epsilon[i, j, t] = P
        gamma[i, t]
        '''
        num_obs = len(obs)
        if num_obs == 0:
            return
        self.init_params(Status, Observations)
        for itr in range(max_iter):
            alpha = self.calc_alpha()
            beta = self.calc_beta()
            epsilon = np.zeros((self.num_s, self.num_s, num_obs))
            gamma = alpha * beta
            gamma /= gamma.sum(axis=0)
            for t in range(num_obs - 1):
                oidx_t1 = self.O2idx[obs[t + 1]]
                for i in range(self.num_s):
                    for j in range(self.num_s):
                        epsilon[i, j, t] = alpha[i, t] * self.A[i, j] * self.B[j, oidx_t1] * beta[j, t + 1]
                tsum = epsilon[:, :, t].sum()
                epsilon[:, :, t] /= tsum




def main():
    hmm = HMM(verbose=True)
    hmm.read_params()
    obs = ['H', 'H', 'T']
    prob_f = hmm.get_prob_of_observation_forward(obs)
    prob_b = hmm.get_prob_of_observation_backward(obs)
    best_status = hmm.get_best_status_sequence(obs)
    print prob_f, prob_b
    print best_status


if __name__ == '__main__':
    main()
