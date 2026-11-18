import numpy as np
import random


def greedy_action(q, state, actions):
    raise NotImplementedError


def epsilon_greedy(q, state, actions, epsilon, rng):
    raise NotImplementedError


def q_learning_update(q, state, action, reward, next_state, actions, alpha, gamma, done):
    raise NotImplementedError


def train_q_learning(env, episodes, alpha, gamma, epsilon, rng):
    raise NotImplementedError


def learning_rate(count):
    raise NotImplementedError


def exploration_value(value, count, r_plus, n_e):
    raise NotImplementedError


def optimistic_action(q, counts, state, actions, r_plus, n_e):
    raise NotImplementedError


def train_optimistic(env, episodes, gamma, r_plus, n_e):
    raise NotImplementedError


def to_array(q, env):
    raise NotImplementedError


def state_values(table):
    raise NotImplementedError


def greedy_policy(table):
    raise NotImplementedError
