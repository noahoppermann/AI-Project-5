# Q-Learning

Implement Q-learning, then the exploring agent from the book, and see how differently the
two of them go about finding the same policy.

## Setup

```
conda env create -f environment.yml
conda activate ai-search
```

## Files

| File | |
| --- | --- |
| `reinforcement_learning.py` | Your code. |
| `main.py` | Trains an agent and draws what it learned. |
| `tests/` | The grading tests. |
| `environment.yml` | The course environment. |
| `env.py` | Test worlds. |

## Tasks

1. Implement `greedy_action` and `epsilon_greedy`.
2. Implement `q_learning_update`.
3. Implement `train_q_learning`.
4. Implement `learning_rate`, `exploration_value`, `optimistic_action` and
   `train_optimistic`.
5. Implement `to_array`, `state_values` and `greedy_policy`, and train on a large world.

## The table

A Q-table is a dictionary keyed by `(state, action)`. A pair that is missing from it has
value `0.0`, so read it with `q.get((state, action), 0.0)` rather than `q[state, action]`.
Both update functions change the table in place and return nothing.

The same table can be held as a numpy array with one row per state and one column per
action. `to_array` builds that array, and `state_values` and `greedy_policy` reduce it to
one number per state. Those two run on tables of half a million rows, so they work on the
whole array at once rather than a row at a time.

## The environment

Use only these four:

| | |
| --- | --- |
| `env.actions` | the actions, always in the same order |
| `env.reset()` | starts a new episode, returns the first state |
| `env.step(action)` | returns `(next_state, reward, done)` |
| `env.is_terminal(state)` | whether the episode is over in that state |

Three more describe the shape of a table:

| | |
| --- | --- |
| `env.states()` | every state, in the order a table's rows use |
| `env.n_states` | how many states there are |
| `env.state_index(state)` | the row this state occupies |

A state is a `(column, row)` pair with `(1, 1)` at the bottom left. Reward is paid on the
move: stepping onto an ordinary square costs a little, stepping onto a terminal square pays
what that square is worth. When `step` returns `done` the episode is over, and calling
`step` again before `reset` is an error.

## Pseudocode

```
function Q-LEARNING(env, episodes, alpha, gamma, epsilon) returns a table Q
    Q <- an empty table of action values, every missing entry counting as 0
    repeat episodes times
        s <- env.RESET()
        repeat
            a <- EPSILON-GREEDY(Q, s, env.ACTIONS, epsilon)
            s', r, done <- env.STEP(a)
            if done then
                Q[s, a] <- Q[s, a] + alpha (r - Q[s, a])
            else
                Q[s, a] <- Q[s, a] + alpha (r + gamma max_a' Q[s', a'] - Q[s, a])
            s <- s'
        until done
    return Q

function EPSILON-GREEDY(Q, s, actions, epsilon) returns an action
    with probability epsilon, return an action drawn uniformly from actions
    otherwise return the action maximising Q[s, a]

function EXPLORING-AGENT(env, episodes, gamma, r+, N_e) returns a table Q and counts N
    Q <- an empty table of action values, every missing entry counting as 0
    N <- an empty table of counts, every missing entry counting as 0
    repeat episodes times
        s <- env.RESET()
        repeat
            a <- the action maximising f(Q[s, a], N[s, a])
            s', r, done <- env.STEP(a)
            N[s, a] <- N[s, a] + 1
            if done then
                Q[s, a] <- Q[s, a] + alpha(N[s, a]) (r - Q[s, a])
            else
                Q[s, a] <- Q[s, a] + alpha(N[s, a]) (r + gamma max_a' Q[s', a'] - Q[s, a])
            s <- s'
        until done
    return Q, N

where alpha(n) = 1 / n and f(u, n) = r+ if n < N_e, and u otherwise
```

The exploring agent is from AIMA Figure 22.8. Instead of exploring at random, `f` reports any pair tried fewer than `N_e`
times as being worth `r+`, an optimistic figure, so the agent goes and tries it; once a pair
has been tried `N_e` times `f` reports what the table actually says. And instead of a fixed
learning rate, a pair's first visit moves its value the whole way to the target and later
visits move it less and less.

It draws no random numbers, so on a world whose moves are reliable the same run happens
every time.

## Requirements

- A pair missing from the table counts as `0.0`.
- `greedy_action` breaks ties by the order of `actions`, so the earliest best action wins.
- `epsilon_greedy` draws its random action uniformly from every action, the greedy one included.
- Draw every random number from the `rng` you are passed, never from `random` directly.
- When `done` is true there is no next state to back up from, so the target is the reward alone.
- Both update functions change the table in place.
- `to_array` puts `q[state, action]` at row `env.state_index(state)` and the column
  `env.actions` holds that action at. Every pair missing from the table is `0.0`.
- `greedy_policy` returns columns, not action names, and breaks ties toward the lowest
  column, which is the rule `greedy_action` already follows.
- Count the visit before you use it, so a pair's first visit has a learning rate of 1.
- `optimistic_action` breaks ties by the order of `actions`, as `greedy_action` does. On an
  empty table every action ties.
- `train_optimistic` returns the pair `(q, counts)`.

## Testing

```
pytest                                 # all tests
pytest -k TestQLearningUpdate          # one part
```

The other four classes are `TestActionSelection`, `TestQLearningTraining`,
`TestOptimisticAgent` and `TestArrayTable`.

## Visualizations

```
python main.py grid       # the policy and values learned on a 4x3 grid
python main.py cliff      # the route Q-learning takes along a cliff
python main.py big        # a 30x30 world, drawn from the array form of the table
python main.py explore    # the exploring agent's policy, and how often it tried each square
```

Each takes `--episodes`, `--gamma` and `--seed`; `grid`, `cliff` and `big` also take
`--alpha` and `--epsilon`, `big` takes `--size`, and `explore` takes `--r-plus` and `--n-e`.

## Submission

Submit `reinforcement_learning.py` through github.

## Grading

| Component | Points |
| --- | --- |
| Action selection | 15 |
| The Q-learning update | 20 |
| Training with Q-learning | 25 |
| Directed exploration | 20 |
| The table as an array | 20 |
| **Total** | **100** |

Code that special-cases a particular state or world earns no credit for that part.
