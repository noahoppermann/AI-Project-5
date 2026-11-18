import os
import random
import sys
import time
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
if os.environ.get("RL_DIR"):
    sys.path.insert(0, os.path.abspath(os.environ["RL_DIR"]))

import reinforcement_learning as rl  # noqa: E402
import env as E                      # noqa: E402

ACTIONS = ("up", "down", "left", "right")
OPTIMAL_GRID_RETURN = 0.84
OPTIMAL_CLIFF_STEPS = 13
EDGE_COLUMNS = range(3, 10)
R_PLUS = 2.0
N_E = 5
ROUTE_POLICY = {(1, 1): {"up", "right"}, (1, 2): {"up"}, (1, 3): {"right"},
                (2, 3): {"right"}, (3, 3): {"right"}}
BESIDE_THE_MINUS_ONE = {((3, 2), "right"): -1.0, ((4, 1), "up"): -1.0}
BIG_SIZE = 30
BIG_EPISODES = 5000
OPTIMAL_BIG_STEPS = 2 * (BIG_SIZE - 1)
BIG_SECONDS = 45
LARGE_TABLE_ROWS = 500_000
TIME_BUDGET = 15


def edge_value(q):
    """Mean value of walking right along the row just above the cliff."""
    vals = [q.get(((c, 2), "right"), 0.0) for c in EDGE_COLUMNS]
    return sum(vals) / len(vals)


class Deadline:
    """A world that gives up instead of hanging, for the large-world test."""

    def __init__(self, world, seconds):
        self._world = world
        self._limit = seconds
        self._start = time.perf_counter()

    def __getattr__(self, name):
        return getattr(self._world, name)

    def reset(self):
        return self._world.reset()

    def step(self, action):
        if time.perf_counter() - self._start > self._limit:
            raise AssertionError(
                f"training on the {BIG_SIZE} by {BIG_SIZE} world passed "
                f"{self._limit} seconds, which the reference does in about a second. "
                f"Something in the loop is doing work proportional to the number of "
                f"states on every step; a Q-learning step should only touch the pairs "
                f"for the state it is in and the state it moved to.")
        return self._world.step(action)


def greedy_walk(world, q, cap=60):
    """Follow the greedy policy from the start. Returns (path, total reward, reached)."""
    state, done, path, total, steps = world.reset(), False, [world.start], 0.0, 0
    while not done and steps < cap:
        action = rl.greedy_action(q, state, world.actions)
        if action not in world.actions:
            raise AssertionError(f"greedy_action returned {action!r}, which is not an action")
        state, reward, done = world.step(action)
        path.append(state)
        total += reward
        steps += 1
    return path, total, world.is_terminal(state)


class RLTest(unittest.TestCase):
    longMessage = False

    def train_q(self, world, episodes=500, alpha=0.5, gamma=1.0, epsilon=0.1, seed=0):
        q = rl.train_q_learning(world, episodes, alpha, gamma, epsilon, random.Random(seed))
        self.assertIsInstance(q, dict, "train_q_learning must return a dict of Q-values; "
                                       f"it returned {type(q).__name__}.")
        return q

class TestActionSelection(RLTest):

    def test_picks_the_best_action(self):
        q = {(("s"), "up"): 1.0, ("s", "down"): 5.0, ("s", "left"): 2.0, ("s", "right"): -1.0}
        got = rl.greedy_action(q, "s", ACTIONS)
        self.assertEqual(got, "down", f"greedy_action returned {got!r}; 'down' has the "
                                      f"highest value (5.0).")

    def test_treats_missing_entries_as_zero(self):
        q = {("s", "up"): -3.0, ("s", "down"): -2.0, ("s", "left"): -5.0}
        got = rl.greedy_action(q, "s", ACTIONS)
        self.assertEqual(got, "right", f"greedy_action returned {got!r}; 'right' is absent "
                                       f"from the table so its value is 0.0, which beats "
                                       f"every value present.")

    def test_breaks_ties_by_action_order(self):
        q = {("s", a): 4.0 for a in ACTIONS}
        got = rl.greedy_action(q, "s", ACTIONS)
        self.assertEqual(got, "up", f"greedy_action returned {got!r}; when every action ties "
                                    f"the first one in actions wins, which is 'up'.")

    def test_epsilon_zero_is_always_greedy(self):
        rng = random.Random(3)
        q = {((c, r), a): rng.uniform(-5, 5) for c in range(1, 5) for r in range(1, 4)
             for a in ACTIONS}
        for state in [(c, r) for c in range(1, 5) for r in range(1, 4)]:
            want = rl.greedy_action(q, state, ACTIONS)
            got = rl.epsilon_greedy(q, state, ACTIONS, 0.0, random.Random(1))
            self.assertEqual(got, want, f"with epsilon=0, epsilon_greedy returned {got!r} at "
                                        f"state {state} but the greedy action is {want!r}.")

    def test_epsilon_one_spreads_over_every_action(self):
        q = {("s", "up"): 100.0}
        rng = random.Random(0)
        counts = {a: 0 for a in ACTIONS}
        draws = 4000
        for _ in range(draws):
            counts[rl.epsilon_greedy(q, "s", ACTIONS, 1.0, rng)] += 1
        missing = [a for a, n in counts.items() if n == 0]
        self.assertFalse(missing, f"with epsilon=1 every action should turn up, but "
                                  f"{missing} never did over {draws} draws: {counts}.")
        low = min(counts.values()) / draws
        self.assertGreater(low, 0.15, f"with epsilon=1 the action should be chosen uniformly, "
                                      f"so each of the 4 should appear about 25% of the time; "
                                      f"the rarest appeared {low:.1%} of the time: {counts}.")

    def test_never_returns_an_illegal_action(self):
        q = {("s", "left"): 2.0}
        rng = random.Random(5)
        for eps in (0.0, 0.3, 1.0):
            for _ in range(200):
                got = rl.epsilon_greedy(q, "s", ACTIONS, eps, rng)
                self.assertIn(got, ACTIONS, f"epsilon_greedy returned {got!r} with "
                                            f"epsilon={eps}, which is not one of {ACTIONS}.")

    def test_uses_the_random_number_generator_it_is_given(self):
        q = {("s", "up"): 1.0}
        a = random.Random(11)
        b = random.Random(11)
        seq_a = [rl.epsilon_greedy(q, "s", ACTIONS, 0.5, a) for _ in range(50)]
        seq_b = [rl.epsilon_greedy(q, "s", ACTIONS, 0.5, b) for _ in range(50)]
        self.assertEqual(seq_a, seq_b, "two generators started from the same seed produced "
                                       "different action sequences, so epsilon_greedy is not "
                                       "drawing from the rng it was passed.")


class TestQLearningUpdate(RLTest):

    def test_update_from_an_empty_table(self):
        q = {}
        rl.q_learning_update(q, "s", "up", 10.0, "t", ACTIONS, 0.5, 0.9, False)
        got = q.get(("s", "up"))
        self.assertIsNotNone(got, "q_learning_update did not write q[('s', 'up')]; it must "
                                  "update the table in place.")
        self.assertAlmostEqual(got, 5.0, places=9,
                               msg=f"q_learning_update gave {got}; starting from 0 with "
                                   f"reward 10, alpha 0.5 and no future value, the new value "
                                   f"is 0 + 0.5 * (10 + 0.9 * 0 - 0) = 5.0.")

    def test_uses_the_best_next_action(self):
        q = {("s", "up"): 1.0, ("t", "up"): 2.0, ("t", "down"): 7.0, ("t", "left"): 3.0}
        rl.q_learning_update(q, "s", "up", 1.0, "t", ACTIONS, 0.5, 1.0, False)
        got = q[("s", "up")]
        self.assertAlmostEqual(got, 4.5, places=9,
                               msg=f"q_learning_update gave {got}; Q-learning bootstraps from "
                                   f"the best next action, which is 7.0, so the new value is "
                                   f"1 + 0.5 * (1 + 1.0 * 7 - 1) = 4.5.")

    def test_a_finished_episode_has_no_future_value(self):
        q = {("s", "up"): 1.0, ("t", "up"): 100.0, ("t", "down"): 100.0,
             ("t", "left"): 100.0, ("t", "right"): 100.0}
        rl.q_learning_update(q, "s", "up", 2.0, "t", ACTIONS, 0.5, 1.0, True)
        got = q[("s", "up")]
        self.assertAlmostEqual(got, 1.5, places=9,
                               msg=f"q_learning_update gave {got}; with done=True there is no "
                                   f"next state to bootstrap from, so the target is the reward "
                                   f"alone and the new value is 1 + 0.5 * (2 - 1) = 1.5.")

    def test_gamma_discounts_the_future(self):
        q = {("s", "up"): 0.0, ("t", "up"): 10.0}
        rl.q_learning_update(q, "s", "up", 0.0, "t", ACTIONS, 1.0, 0.5, False)
        got = q[("s", "up")]
        self.assertAlmostEqual(got, 5.0, places=9,
                               msg=f"q_learning_update gave {got}; with gamma 0.5 the future "
                                   f"value 10 counts half, so the new value is 5.0.")

    def test_alpha_zero_changes_nothing(self):
        q = {("s", "up"): 3.0, ("t", "up"): 99.0}
        rl.q_learning_update(q, "s", "up", 50.0, "t", ACTIONS, 0.0, 1.0, False)
        got = q[("s", "up")]
        self.assertAlmostEqual(got, 3.0, places=9,
                               msg=f"q_learning_update gave {got}; with alpha=0 nothing is "
                                   f"learned and the value stays 3.0.")

    def test_alpha_one_replaces_the_value(self):
        q = {("s", "up"): 3.0, ("t", "up"): 10.0}
        rl.q_learning_update(q, "s", "up", 1.0, "t", ACTIONS, 1.0, 1.0, False)
        got = q[("s", "up")]
        self.assertAlmostEqual(got, 11.0, places=9,
                               msg=f"q_learning_update gave {got}; with alpha=1 the old value "
                                   f"is thrown away and replaced by 1 + 1.0 * 10 = 11.0.")

    def test_changes_only_the_pair_it_was_given(self):
        q = {("s", "up"): 1.0, ("s", "down"): 2.0, ("t", "up"): 3.0}
        before = dict(q)
        rl.q_learning_update(q, "s", "up", 5.0, "t", ACTIONS, 0.5, 1.0, False)
        for key, was in before.items():
            if key != ("s", "up"):
                self.assertAlmostEqual(q[key], was, places=9,
                                       msg=f"q_learning_update changed q[{key}] from {was} to "
                                           f"{q[key]}; only the pair it is given may change.")


class TestQLearningTraining(RLTest):

    def test_returns_a_table_of_values(self):
        q = self.train_q(E.simple_world(rng=random.Random(1)), episodes=50)
        self.assertTrue(q, "train_q_learning returned an empty table after 50 episodes.")
        for key, value in q.items():
            self.assertIsInstance(key, tuple, f"table keys should be (state, action) pairs; "
                                              f"found the key {key!r}.")
            self.assertEqual(len(key), 2, f"table keys should be (state, action) pairs; "
                                          f"found the key {key!r}.")
            self.assertIsInstance(value, float, f"table values should be numbers; "
                                                f"q[{key!r}] is {value!r}.")

    def test_finds_the_best_route_through_the_grid(self):
        q = self.train_q(E.simple_world(rng=random.Random(11)))
        path, total, reached = greedy_walk(E.simple_world(rng=random.Random(2)), q)
        self.assertTrue(reached, f"after training, the greedy policy never reached a terminal "
                                 f"square; it walked {path[:12]}...")
        self.assertAlmostEqual(total, OPTIMAL_GRID_RETURN, places=6,
                               msg=f"the greedy policy collected {total:.2f} walking from the "
                                   f"start, but the best possible is {OPTIMAL_GRID_RETURN}. "
                                   f"It took the route {path}.")

    def test_reaches_the_goal_on_the_cliff(self):
        q = self.train_q(E.CliffWalk(rng=random.Random(11)))
        path, total, reached = greedy_walk(E.CliffWalk(rng=random.Random(2)), q)
        self.assertTrue(reached, f"after training, the greedy policy never reached the goal; "
                                 f"it walked {path[:12]}...")

    def test_takes_the_shortest_route_on_the_cliff(self):
        q = self.train_q(E.CliffWalk(rng=random.Random(11)))
        path, total, reached = greedy_walk(E.CliffWalk(rng=random.Random(2)), q)
        self.assertTrue(reached, "the greedy policy never reached the goal on the cliff.")
        self.assertEqual(len(path) - 1, OPTIMAL_CLIFF_STEPS,
                         f"the greedy policy reached the goal in {len(path) - 1} steps; the "
                         f"shortest route is {OPTIMAL_CLIFF_STEPS}. It walked {path}.")

    def test_walks_the_row_beside_the_cliff(self):
        q = self.train_q(E.CliffWalk(rng=random.Random(11)))
        path, total, reached = greedy_walk(E.CliffWalk(rng=random.Random(2)), q)
        self.assertTrue(reached, "the greedy policy never reached the goal on the cliff.")
        highest = max(row for _, row in path)
        self.assertLessEqual(highest, 2, f"the greedy policy climbed to row {highest}. "
                                         f"Q-learning values the shortest route, which runs "
                                         f"along row 2 beside the cliff. It walked {path}.")

    def test_learns_a_thirty_by_thirty_world(self):
        world = Deadline(E.big_world(BIG_SIZE, rng=random.Random(11)), BIG_SECONDS)
        q = self.train_q(world, episodes=BIG_EPISODES)
        path, total, reached = greedy_walk(E.big_world(BIG_SIZE, rng=random.Random(2)), q,
                                           cap=10 * OPTIMAL_BIG_STEPS)
        self.assertTrue(reached, f"after {BIG_EPISODES} episodes on the {BIG_SIZE} by "
                                 f"{BIG_SIZE} world the greedy policy never reached the "
                                 f"goal; it walked {path[:12]}...")
        self.assertEqual(len(path) - 1, OPTIMAL_BIG_STEPS,
                         f"the greedy policy reached the goal in {len(path) - 1} steps; "
                         f"every shortest route takes {OPTIMAL_BIG_STEPS}. The world is "
                         f"open, so any route that only goes up and right is shortest.")

    def test_values_the_cliff_edge_at_its_best(self):
        q = self.train_q(E.CliffWalk(rng=random.Random(11)))
        got = edge_value(q)
        self.assertGreater(got, -10.0,
                           f"the mean value of walking right along row 2 came out {got:.2f}. "
                           f"Q-learning learns the value of the best route regardless of the "
                           f"exploring it does, so this should be close to -7.")


class TestArrayTable(RLTest):

    def small(self):
        return E.simple_world(rng=random.Random(1))

    def test_builds_one_row_per_state_and_one_column_per_action(self):
        world = self.small()
        q = {((1, 1), "right"): 2.5, ((3, 2), "up"): -1.25}
        table = rl.to_array(q, world)
        self.assertIsInstance(table, np.ndarray, "to_array must return a numpy array; it "
                                                 f"returned {type(table).__name__}.")
        self.assertEqual(table.shape, (world.n_states, len(world.actions)),
                         f"to_array returned an array of shape {table.shape}; it should be "
                         f"({world.n_states}, {len(world.actions)}), one row per state in "
                         f"env.states() order and one column per action in env.actions order.")
        got = table[world.state_index((1, 1)), world.actions.index("right")]
        self.assertAlmostEqual(got, 2.5, places=9,
                               msg=f"q[(1, 1), 'right'] is 2.5 but the matching cell of the "
                                   f"array holds {got}. Row env.state_index(state), column "
                                   f"env.actions.index(action).")
        got = table[world.state_index((3, 2)), world.actions.index("up")]
        self.assertAlmostEqual(got, -1.25, places=9,
                               msg=f"q[(3, 2), 'up'] is -1.25 but the matching cell holds "
                                   f"{got}.")

    def test_pairs_missing_from_the_table_are_zero(self):
        world = self.small()
        table = rl.to_array({((1, 1), "up"): 4.0}, world)
        self.assertAlmostEqual(table.sum(), 4.0, places=9,
                               msg=f"the table holds one value, 4.0, so the array should sum "
                                   f"to 4.0; it sums to {table.sum()}. Every pair missing "
                                   f"from the table is 0.0.")

    def test_state_values_takes_the_best_in_each_row(self):
        table = np.array([[1.0, 5.0, 2.0, -1.0],
                          [0.0, 0.0, 0.0, 0.0],
                          [-3.0, -3.0, -4.0, -9.0],
                          [7.5, 7.5, 7.5, 7.5]])
        got = rl.state_values(table)
        self.assertIsInstance(got, np.ndarray, "state_values must return a numpy array; it "
                                               f"returned {type(got).__name__}.")
        np.testing.assert_allclose(
            got, [5.0, 0.0, -3.0, 7.5],
            err_msg=f"state_values returned {np.asarray(got).tolist()}; each entry is the "
                    f"largest value in its row.")

    def test_greedy_policy_takes_the_best_column(self):
        table = np.array([[1.0, 5.0, 2.0, -1.0],
                          [0.0, 0.0, 0.0, 0.0],
                          [-3.0, -3.0, -4.0, -9.0],
                          [-1.0, -2.0, -3.0, -4.0]])
        got = rl.greedy_policy(table)
        self.assertIsInstance(got, np.ndarray, "greedy_policy must return a numpy array; it "
                                               f"returned {type(got).__name__}.")
        np.testing.assert_array_equal(
            np.asarray(got), [1, 0, 0, 0],
            err_msg=f"greedy_policy returned {np.asarray(got).tolist()}; each entry is the "
                    f"column of the largest value in its row, and a tie goes to the lowest "
                    f"column, as greedy_action breaks ties by the order of actions.")

    def test_the_array_agrees_with_the_table_it_came_from(self):
        world = E.big_world(10, rng=random.Random(11))
        q = self.train_q(world, episodes=400)
        table = rl.to_array(q, world)
        values, policy = rl.state_values(table), rl.greedy_policy(table)
        for state in world.states():
            row = world.state_index(state)
            best = max(q.get((state, a), 0.0) for a in world.actions)
            self.assertAlmostEqual(
                values[row], best, places=9,
                msg=f"state {state} is row {row}. Its best action value in the table is "
                    f"{best:.4f}, but state_values gives {values[row]:.4f}.")
            chosen = world.actions[policy[row]]
            self.assertAlmostEqual(
                q.get((state, chosen), 0.0), best, places=9,
                msg=f"at state {state}, greedy_policy picked column {policy[row]} "
                    f"({chosen!r}), worth {q.get((state, chosen), 0.0):.4f}, but the best "
                    f"available is {best:.4f}.")

    def test_summarises_a_large_table_without_looping(self):
        table = np.random.default_rng(0).normal(size=(LARGE_TABLE_ROWS, 4))
        baseline = min(self.timed(lambda: table.sum(axis=1)) for _ in range(3))
        allowed = TIME_BUDGET * baseline
        for name, f in (("state_values", rl.state_values), ("greedy_policy", rl.greedy_policy)):
            taken = min(self.timed(lambda: f(table)) for _ in range(2))
            self.assertLess(
                taken, allowed,
                f"{name} took {taken * 1000:.0f} ms on a table of {LARGE_TABLE_ROWS:,} rows. "
                f"One numpy pass over the same table takes {baseline * 1000:.0f} ms, and the "
                f"budget is {TIME_BUDGET} times that, {allowed * 1000:.0f} ms. Work on the "
                f"whole array at once rather than a row at a time.")

    @staticmethod
    def timed(f):
        start = time.perf_counter()
        f()
        return time.perf_counter() - start


class TestOptimisticAgent(RLTest):

    def train_o(self, world, episodes=500, gamma=1.0):
        out = rl.train_optimistic(world, episodes, gamma, R_PLUS, N_E)
        self.assertIsInstance(out, tuple, "train_optimistic must return the pair "
                                          f"(q, counts); it returned {type(out).__name__}.")
        self.assertEqual(len(out), 2, f"train_optimistic must return the pair (q, counts); "
                                      f"it returned {len(out)} things.")
        return out

    def test_the_learning_rate_is_one_over_the_visit_count(self):
        for count, want in ((1, 1.0), (2, 0.5), (4, 0.25), (10, 0.1)):
            got = rl.learning_rate(count)
            self.assertAlmostEqual(got, want, places=9,
                                   msg=f"learning_rate({count}) returned {got}; it should be "
                                       f"1 / {count} = {want}. The first visit to a pair moves "
                                       f"its value all the way to the target, and later visits "
                                       f"move it less and less.")

    def test_exploration_value_prefers_an_untried_pair(self):
        got = rl.exploration_value(-50.0, 0, R_PLUS, N_E)
        self.assertAlmostEqual(got, R_PLUS, places=9,
                               msg=f"exploration_value(-50.0, 0, {R_PLUS}, {N_E}) returned "
                                   f"{got}; a pair tried fewer than n_e times is worth r_plus, "
                                   f"whatever the table says about it.")
        got = rl.exploration_value(-50.0, N_E, R_PLUS, N_E)
        self.assertAlmostEqual(got, -50.0, places=9,
                               msg=f"exploration_value(-50.0, {N_E}, {R_PLUS}, {N_E}) returned "
                                   f"{got}; once a pair has been tried n_e times it is worth "
                                   f"what the table says, -50.0.")

    def test_optimistic_action_tries_a_new_action_before_a_good_one(self):
        q = {("s", "up"): 1.9, ("s", "down"): 1.8}
        counts = {("s", "up"): 100, ("s", "down"): 100, ("s", "left"): 100}
        got = rl.optimistic_action(q, counts, "s", ACTIONS, R_PLUS, N_E)
        self.assertEqual(got, "right", f"optimistic_action returned {got!r}; 'right' has never "
                                       f"been tried, so it is worth r_plus ({R_PLUS}), which "
                                       f"beats the 1.9 on offer from 'up'.")

    def test_optimistic_action_breaks_ties_by_action_order(self):
        got = rl.optimistic_action({}, {}, "s", ACTIONS, R_PLUS, N_E)
        self.assertEqual(got, ACTIONS[0], f"optimistic_action returned {got!r}; with an empty "
                                          f"table every action is worth r_plus, and a tie goes "
                                          f"to the first action in actions, {ACTIONS[0]!r}.")

    def test_every_pair_is_tried_at_least_n_e_times(self):
        world = E.simple_world(rng=random.Random(7))
        q, counts = self.train_o(world)
        thin = {(s, a): counts.get((s, a), 0) for s in world.states()
                for a in world.actions
                if not world.is_terminal(s) and counts.get((s, a), 0) < N_E}
        self.assertFalse(thin, f"after 500 episodes these pairs had been tried fewer than "
                               f"{N_E} times: {thin}. The exploration function is what "
                               f"guarantees coverage: a pair below n_e looks worth r_plus, so "
                               f"the agent goes and tries it.")

    def test_finds_the_best_route_through_the_grid(self):
        q, counts = self.train_o(E.simple_world(rng=random.Random(7)))
        world = E.simple_world(rng=random.Random(3))
        path, total, reached = greedy_walk(world, q)
        self.assertTrue(reached, f"after training, the greedy policy never reached a terminal "
                                 f"square; it walked {path[:12]}...")
        self.assertAlmostEqual(total, OPTIMAL_GRID_RETURN, places=6,
                               msg=f"the greedy policy collected {total:.2f} from the start, "
                                   f"but the best possible is {OPTIMAL_GRID_RETURN}. It took "
                                   f"the route {path}.")
        wrong = {}
        for state, best in sorted(ROUTE_POLICY.items()):
            chosen = rl.greedy_action(q, state, world.actions)
            if chosen not in best:
                wrong[state] = chosen
        self.assertFalse(wrong, f"along the route the greedy policy differs from the one value "
                                f"iteration gives, at these states: {wrong}. These are the "
                                f"squares the agent passes through often, so their values "
                                f"should be well settled.")

    def test_learns_to_keep_clear_of_the_minus_one(self):
        q, counts = self.train_o(E.simple_world(rng=random.Random(7)))
        for (state, action), want in sorted(BESIDE_THE_MINUS_ONE.items()):
            got = q.get((state, action), 0.0)
            self.assertAlmostEqual(
                got, want, places=9,
                msg=f"q[{state}, {action!r}] came out {got:.4f}, but that move steps straight "
                    f"onto the -1 square, so the value should be exactly {want}. The first "
                    f"visit to a pair has a learning rate of 1, and the target for a move "
                    f"that ends the episode is the reward alone.")
            best = rl.greedy_action(q, state, E.GridWorld.actions)
            self.assertNotEqual(best, action, f"the greedy action at {state} is {action!r}, "
                                              f"which steps onto the -1 square.")

    def test_training_needs_no_random_numbers(self):
        first = self.train_o(E.simple_world(rng=random.Random(1)), episodes=300)
        second = self.train_o(E.simple_world(rng=random.Random(99)), episodes=300)
        self.assertEqual(first[1], second[1],
                         "two runs on the same deterministic world gave different visit "
                         "counts. The exploring agent draws no random numbers at all: which "
                         "action it tries next follows from the table and the counts, so the "
                         "run is reproducible.")
        self.assertEqual(first[0], second[0],
                         "two runs on the same deterministic world gave different tables, "
                         "though the agent draws no random numbers.")


if __name__ == "__main__":
    unittest.main()
