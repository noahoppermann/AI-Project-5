"""Environments. Do not modify.

Every environment offers the same four things:

    env.actions                 the available actions, always in the same order
    env.reset()                 start a new episode, returns the first state
    env.step(action)            returns (next_state, reward, done)
    env.is_terminal(state)      whether the game is over in that state
    env.states()                every state, in the order a table's rows use
    env.n_states                how many states there are
    env.state_index(state)      the row this state occupies in such a table

A reward is paid on the transition: moving to an ordinary square costs
step_reward, and moving onto a terminal square pays that square's value.
An episode that runs past max_steps is cut off with done set to True.
"""
import random

UP, DOWN, LEFT, RIGHT = "up", "down", "left", "right"
DELTA = {UP: (0, 1), DOWN: (0, -1), LEFT: (-1, 0), RIGHT: (1, 0)}
PERPENDICULAR = {UP: (LEFT, RIGHT), DOWN: (LEFT, RIGHT),
                 LEFT: (DOWN, UP), RIGHT: (DOWN, UP)}


class _Episode:
    """Shared bookkeeping so a runaway episode stops instead of hanging."""

    max_steps = 5000
    max_total_steps = 500_000
    _index = None
    _total = 0

    def _begin(self):
        self._steps = 0
        self._over = False

    @property
    def n_states(self):
        """How many states the world has, so a table can hold one row each."""
        self._ensure_index()
        return len(self._index)

    def state_index(self, state):
        """The row this state occupies in a table with one row per state."""
        self._ensure_index()
        return self._index[state]

    def _ensure_index(self):
        if self._index is None:
            self._index = {s: i for i, s in enumerate(self.states())}

    def _check(self):
        if self._over:
            raise RuntimeError(
                "step() was called after the episode ended. Stop the episode when "
                "step() returns done=True, then call reset() to start another.")
        self._steps += 1
        self._total += 1
        if self._total > self.max_total_steps:
            raise RuntimeError(
                f"this world has taken over {self.max_total_steps:,} steps in total, far "
                "more than reaching the goal needs. Get the small worlds right first.")
        return self._steps >= self.max_steps


class GridWorld(_Episode):
    """A rectangular grid. A state is a (column, row) pair with (1, 1) bottom left.

    With slip > 0 the agent moves as intended with probability 1 - 2 * slip, and
    perpendicular to its intended direction with probability slip on each side.
    Walking into a wall or off the edge leaves the agent where it was.
    """

    actions = (UP, DOWN, LEFT, RIGHT)

    def __init__(self, width, height, terminals, walls=(), start=(1, 1),
                 step_reward=-0.04, slip=0.0, rng=None):
        self.width, self.height = width, height
        self.terminals = dict(terminals)
        self.walls = set(walls)
        self.start = start
        self.step_reward = step_reward
        self.slip = slip
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        self._begin()
        self.state = self.start
        return self.state

    def step(self, action):
        cut = self._check()
        if action not in self.actions:
            raise ValueError(f"{action!r} is not a legal action; use env.actions")
        taken = action
        if self.slip:
            roll = self.rng.random()
            left, right = PERPENDICULAR[action]
            if roll < self.slip:
                taken = left
            elif roll < 2 * self.slip:
                taken = right
        self.state = self._move(self.state, taken)
        done = self.is_terminal(self.state)
        reward = self.terminals[self.state] if done else self.step_reward
        self._over = done or cut
        return self.state, reward, self._over

    def is_terminal(self, state):
        return state in self.terminals

    def states(self):
        return [(c, r) for c in range(1, self.width + 1)
                for r in range(1, self.height + 1) if (c, r) not in self.walls]

    def _move(self, state, action):
        dc, dr = DELTA[action]
        nxt = (state[0] + dc, state[1] + dr)
        if nxt in self.walls or not (1 <= nxt[0] <= self.width
                                     and 1 <= nxt[1] <= self.height):
            return state
        return nxt


class CliffWalk(_Episode):
    """Cliff walking, 12 wide and 4 tall, from Sutton and Barto.

    Every step costs -1. Stepping onto the cliff along the bottom row costs -100
    and puts the agent back at the start without ending the episode. Moves are
    deterministic.
    """

    actions = (UP, DOWN, LEFT, RIGHT)

    def __init__(self, rng=None):
        self.width, self.height = 12, 4
        self.start, self.goal = (1, 1), (12, 1)
        self.cliff = {(c, 1) for c in range(2, 12)}
        self.rng = rng or random.Random()
        self.reset()

    def reset(self):
        self._begin()
        self.state = self.start
        return self.state

    def step(self, action):
        cut = self._check()
        if action not in self.actions:
            raise ValueError(f"{action!r} is not a legal action; use env.actions")
        dc, dr = DELTA[action]
        nxt = (self.state[0] + dc, self.state[1] + dr)
        if not (1 <= nxt[0] <= self.width and 1 <= nxt[1] <= self.height):
            nxt = self.state
        if nxt in self.cliff:
            self.state = self.start
            self._over = cut
            return self.state, -100.0, self._over
        self.state = nxt
        done = self.is_terminal(self.state)
        self._over = done or cut
        return self.state, -1.0, self._over

    def is_terminal(self, state):
        return state == self.goal

    def states(self):
        return [(c, r) for c in range(1, self.width + 1)
                for r in range(1, self.height + 1)]


def simple_world(rng=None):
    """A 4x3 grid with a +1 and a -1 square. Moves always go where they are aimed."""
    return GridWorld(4, 3, terminals={(4, 3): 1.0, (4, 2): -1.0}, walls=[(2, 2)],
                     start=(1, 1), step_reward=-0.04, slip=0.0, rng=rng)


def slippery_world(rng=None):
    """The same grid, but each move slips sideways one time in five."""
    return GridWorld(4, 3, terminals={(4, 3): 1.0, (4, 2): -1.0}, walls=[(2, 2)],
                     start=(1, 1), step_reward=-0.04, slip=0.1, rng=rng)


def big_world(size=10, slip=0.0, rng=None):
    """An open square grid with the goal in the far corner. Every step costs -1.

    The shortest route is 2 * (size - 1) steps, so the best return is -2 * (size - 1).
    At size 30 the world holds 900 states and 3,600 Q-values.
    """
    world = GridWorld(size, size, terminals={(size, size): -1.0}, walls=[],
                      start=(1, 1), step_reward=-1.0, slip=slip, rng=rng)
    world.max_steps = 10_000
    world.max_total_steps = 5_000_000
    return world
