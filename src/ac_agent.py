import numpy as np
from fourier import FourierBasis
import pdb
import os
from datetime import datetime
import copy
from scipy.special import expit, logsumexp
import time

from policy import SoftmaxPolicy


class Actor(object):
    def __init__(self, rng, n_actions, n_obs, n_features, lr_theta=0.001, temperature=1.0):
        self.rng = rng
        self.n_actions = n_actions
        self.n_features = n_features
        self.theta = np.zeros((n_actions, self.n_features))
        self.lr_theta = lr_theta
        self.temperature = temperature
        
    def update(self, feat, action, q_value):
        lr = self.lr_theta/np.linalg.norm(feat)
        self.theta += lr * q_value * self.grad(feat)

    def act(self, feat):
        return int(self.rng.choice(self.weights.shape[1], p=self.pmf(phi)))

    def pmf(self, feat):
        """
        Returns: ndarray(#actions)
        """
        # TODO check
        energy = self.value(self.theta, feat) # / self.temperature # >> (1, #actions)
        return np.exp(energy - logsumexp(energy))
    
    def value(self, feat, action=None):
        energy = np.dot(self.theta, feat)
        if action is None:
            return energy
        return energy[action]
    
    def grad(self, feat):
        action_dist = self.pmf(feat)
        action_dist = action_dist.reshape(1, len(action_dist))
        feat = feat.reshape(1, len(feat))
        grad = np.multiply(action_dist.T, feat) # 
        grad[a] += feat
        return grad


class Critic(object):
    def __init__(self, n_actions, n_features, gamma, lr_q):
        self.w_q = np.zeros((n_actions, n_features))
        self.gamma = gamma
        self.lr_q = lr_q

    def update(self, feat, action, reward, next_feat, next_action):
        # Q値の算出はfeatと重みの内積で算出する．
        td_error = reward + self.gamma * self.value(next_feat, next_action) - self.value(feat, action)
        self.w_q[:] += self.lr_q * td_error * self.grad(feat, action)

    def value(self, feat, action=None):
        if action is None:
            return np.dot(self.w_q[:], feat)
        return np.dot(self.w_q[action], feat)

    def grad(self, feat, action):
        # TODO Q(s,a) = w_a * φ_a(s)
        # 本来なら．該当actionだけfeatでそれ以外は0．
        grad = np.zeros(self.w_q.shape)
        grad[action] = feat
        return feat


class ActorCrticiAgent(object):
    def __init__(self, action_space, observation_space, basis_order=3, epsilon=0.01, gamma=0.99, lr_critic=0.01, lr_q=0.01):
        self.action_space = action_space
        self.basis_order = basis_order
        self.shape_state = observation_space.shape # case Pinball Box(4,0)
        self.fourier_basis = FourierBasis(self.shape_state[0], observation_space, order=self.basis_order)
        self.n_features = self.fourier_basis.getNumBasisFunctions()
        self.rng = np.random.seed(seed = 32)

        # parameters
        # Hyper parameters
        self.epsilon = epsilon
        self.lr_critic = lr_critic
        self.actor = Actor(self.rng, action_space.n, self.n_features)
        self.critic = Critic(action_space.n, self.n_features, gamma, lr_q)

        # variables for analysis
        self.td_error_list = []
        self.td_error_list_meta = []
        self.vis_action_dist = np.zeros(action_space.n)
        self.vis_option_q = np.zeros(n_options)

    def act(self, observation):
        feature = self.fourier_basis(observation)
        self.vis_action_dist = self.actor.pmf(feat)
        return self.actor.act(feat)[0]

    def update(self, pre_obs, pre_a, r, obs, a, done):
        """
        1. Update critic(pi_Omega: Intra Q Learning, pi_u: IntraAction Q learning)
        2. Improve actors
        """
        pre_feat = self.fourier_basis(pre_obs)
        feat = self.fourier_basis(obs)
        q_value = self.critic.value(feat)
        self.actor.update(feat, a, q_value)
        self.critic.update(feat, pre_feat, pre_a, r, obs, a)
        # TODO this is baseline version.
        # q_u_list -= q_omega
