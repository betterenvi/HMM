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

    def init_params(self, Status = None, Observations = None):
        '''
        for Baum_Welch_algorithm
        '''
        if Status: # may already exist in self
            self.Status = copy.copy(Status)
        if Observations:
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
                alpha[j, t] = (alpha[:, t - 1] * self.A[:, j]).sum() * self.B[j, oidx]
        self.alpha = alpha
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
                beta[i, t] = (self.A[i, :] * self.B[:, oidx_t1] * beta[:, t + 1]).sum()
        self.beta = beta
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
                tmp = vertibi[:, t - 1] * self.A[:, j]
                best_prev = tmp.argmax()
                prev[j, t] = best_prev
                vertibi[j, t] = tmp[best_prev] * self.B[j, oidx]
        self.vertibi = vertibi
        self.prev = prev
        return vertibi, prev

    def get_prob_of_observation_forward(self, obs, calc=True):
        num_obs = len(obs)
        if num_obs == 0:
            return None
        if calc:
            self.calc_alpha(obs)
        if self.verbose:
            print '\nalpha:\n', self.alpha
        return self.alpha[:, num_obs - 1].sum()

    def get_prob_of_observation_backward(self, obs, calc=True):
        num_obs = len(obs)
        if num_obs == 0:
            return None
        if calc:
            self.calc_beta(obs)
        if self.verbose:
            print '\nbeta:\n', self.beta
        return (self.pi * self.B[:, self.O2idx[obs[0]]] * self.beta[:, 0]).sum()

    def get_best_status_sequence(self, obs, calc=True):
        best_status = []
        num_obs = len(obs)
        if num_obs == 0:
            return best_status
        if calc:
            self.calc_vertibi_and_prev(obs)
        tmp_status = np.argmax(self.vertibi[:, -1])
        best_status.append(tmp_status)
        for t in range(num_obs - 1, 0, -1):
            tmp_status = self.prev[int(tmp_status), t]
            best_status.append(tmp_status)
        best_status = best_status[::-1]
        if self.verbose:
            print '\nvertibi:\n', self.vertibi
            print '\nprev:\n', self.prev
        return self.Status[best_status]

    def Baum_Welch_algorithm(self, obs, Status = None, Observations = None, max_iter = 100, min_change = 1e-5):
        '''
        epsilon[i, j, t] = P
        gamma[i, t]
        '''
        num_obs = len(obs)
        if num_obs == 0:
            return
        self.init_params(Status, Observations)
        tarr = np.zeros((self.num_o, num_obs))
        tarr[:, :] = np.array(range(self.num_o))[:, np.newaxis]
        sigma = (tarr == np.array(self.O2idx[obs])) + 0
        if self.verbose:
            print '\nsigma:\n', sigma
        log_prob_o_given_lambda, log_prob_o_given_lambda_old = None, None
        for itr in range(max_iter):
            alpha = self.calc_alpha(obs)
            beta = self.calc_beta(obs)
            epsilon = np.zeros((self.num_s, self.num_s, num_obs))
            gamma = alpha * beta
            gamma /= gamma.sum(axis=0)
            for t in range(num_obs - 1):
                oidx_t1 = self.O2idx[obs[t + 1]]
                epsilon[:, :, t] = alpha[:, t][:, np.newaxis]
                for i in range(self.num_s):
                    epsilon[i, :, t] *= self.A[i, :] * self.B[:, oidx_t1] * beta[:, t + 1]
                tsum = epsilon[:, :, t].sum()
                epsilon[:, :, t] /= tsum
            status_expectation = gamma.sum(axis=1)
            status_from_expectation = status_expectation - gamma[:, -1]
            # pi_i = gamma[i, 0]
            self.pi = gamma[:, 0]
            # a_{ij} = \frac{\Sigma_{t=1}^{T-1} epsilon[i, j, t]}{\Sigma_{t=1}^{T-1} gamma[i, t]}
            self.A = epsilon[:, :, :-1].sum(axis=2) / status_from_expectation[:, np.newaxis]
            # b_{jk} = \frac{\Sigma_{t=1}^{T} gamma[j, t] * \sigma(o_t, v_k)}{\Sigma_{t=1}^{T} gamma[j, t]}, where \sigma(o_t, v_k) = 1 if o_t == v_k else 0
            self.B = np.dot(gamma, sigma.T) / status_expectation[:, np.newaxis]
            if self.verbose:
                print '\n\n*** iter', itr, '***\n'
                print '\nalpha:\n', alpha
                print '\nbeta:\n', beta
                print '\nepsilon:\n', epsilon
                print '\nsigma:\n', gamma
                print '\npi:\n', self.pi
                print '\nA:\n', self.A
                print '\nB:\n', self.B
            log_prob_o_given_lambda = np.log(self.get_prob_of_observation_forward(obs, calc=False))
            if log_prob_o_given_lambda_old and abs(log_prob_o_given_lambda - log_prob_o_given_lambda_old) < min_change:
                break
            log_prob_o_given_lambda_old = log_prob_o_given_lambda


        print '\npi:\n', self.pi
        print '\nA:\n', self.A
        print '\nB:\n', self.B


def main():
    hmm = HMM(verbose=False)
    hmm.read_params()
    obs = ['H', 'H', 'T']
    prob_f = hmm.get_prob_of_observation_forward(obs)
    prob_b = hmm.get_prob_of_observation_backward(obs)
    best_status = hmm.get_best_status_sequence(obs)
    print prob_f, prob_b
    print list(best_status)
    # test Baum_Welch_algorithm
    hmm.Baum_Welch_algorithm(obs, max_iter=10, min_change=1e-5)

if __name__ == '__main__':
    main()
