import argparse
import random
import time

import env as E
import reinforcement_learning as rl

ARROW = {"up": "^", "down": "v", "left": "<", "right": ">"}


def show_grid(world, q):
    for row in range(world.height, 0, -1):
        marks, values = [], []
        for col in range(1, world.width + 1):
            s = (col, row)
            if s in world.walls:
                marks.append("  #  ")
                values.append("     ")
            elif world.is_terminal(s):
                marks.append(f"{world.terminals[s]:+.0f}".center(5))
                values.append(f"{world.terminals[s]:+.2f}")
            else:
                a = rl.greedy_action(q, s, world.actions)
                marks.append(ARROW[a].center(5))
                values.append(f"{q.get((s, a), 0.0):+.2f}")
        print("   " + " ".join(marks) + "    " + " ".join(values))


def run_grid(args):
    for name, make in [("moves always go where aimed", E.simple_world),
                       ("moves slip sideways one time in five", E.slippery_world)]:
        world = make(rng=random.Random(args.seed + 1))
        q = rl.train_q_learning(world, args.episodes, args.alpha, args.gamma,
                                args.epsilon, random.Random(args.seed))
        print(f"\n{name}   ({args.episodes} episodes)")
        print("   policy" + " " * 22 + "value of the chosen action")
        show_grid(world, q)


def greedy_walk(q, cap=60):
    world = E.CliffWalk(rng=random.Random(0))
    state, done, path, steps, total = world.reset(), False, [world.start], 0, 0.0
    while not done and steps < cap:
        state, reward, done = world.step(rl.greedy_action(q, state, world.actions))
        path.append(state)
        total += reward
        steps += 1
    return path, total, state == world.goal


def show_walk(path):
    on_path = set(path)
    world = E.CliffWalk()
    for row in range(world.height, 0, -1):
        cells = []
        for col in range(1, world.width + 1):
            s = (col, row)
            if s == world.start:
                cells.append("S")
            elif s == world.goal:
                cells.append("G")
            elif s in world.cliff:
                cells.append("!")
            else:
                cells.append("*" if s in on_path else ".")
        print("   " + " ".join(cells))


def behaviour_return(q, epsilon, episodes, seed):
    """What the exploring policy actually earns, as opposed to what it learned."""
    totals = []
    for i in range(episodes):
        world = E.CliffWalk(rng=random.Random(seed + i))
        rng = random.Random(seed + 1000 + i)
        state, done, total = world.reset(), False, 0.0
        while not done:
            state, reward, done = world.step(
                rl.epsilon_greedy(q, state, world.actions, epsilon, rng))
            total += reward
        totals.append(total)
    return sum(totals) / len(totals)


def run_cliff(args):
    print("S start   G goal   ! cliff   * the route the greedy policy takes\n")
    world = E.CliffWalk(rng=random.Random(args.seed + 1))
    q = rl.train_q_learning(world, args.episodes, args.alpha, args.gamma, args.epsilon,
                            random.Random(args.seed))
    path, total, reached = greedy_walk(q)
    print(f"Q-learning   ({args.episodes} episodes)")
    show_walk(path)
    if not reached:
        print("   did not reach the goal")
        return
    edge = sum(q.get(((c, 2), "right"), 0.0) for c in range(3, 10)) / 7
    behaving = behaviour_return(q, args.epsilon, 200, args.seed + 50)
    print(f"   greedy policy: reached the goal in {len(path) - 1} steps, return {total:+.0f}")
    print(f"   mean value of walking right along the cliff edge: {edge:+.2f}")
    print(f"   same policy with epsilon still at {args.epsilon}: {behaving:+.1f} a run\n")
    print("   Q-learning backs up the best action available next, not the one it is about")
    print("   to take, so it learns the value of the cliff-edge route even while its own")
    print(f"   exploring keeps knocking it off. Following that route at epsilon "
          f"{args.epsilon} costs")
    print(f"   {behaving:+.1f} a run; the route itself is worth {total:+.0f}.")


def show_policy(world, policy):
    for row in range(world.height, 0, -1):
        cells = []
        for col in range(1, world.width + 1):
            state = (col, row)
            if world.is_terminal(state):
                cells.append("G")
            elif state == world.start:
                cells.append("S")
            else:
                cells.append(ARROW[world.actions[policy[world.state_index(state)]]])
        print("   " + "".join(cells))


def steps_to_goal(world, q, cap):
    state, done, steps = world.reset(), False, 0
    while not done and steps < cap:
        state, reward, done = world.step(rl.greedy_action(q, state, world.actions))
        steps += 1
    return steps if world.is_terminal(state) else None


def run_big(args):
    world = E.big_world(args.size, rng=random.Random(args.seed + 1))
    start = time.perf_counter()
    q = rl.train_q_learning(world, args.episodes, args.alpha, args.gamma, args.epsilon,
                            random.Random(args.seed))
    taken = time.perf_counter() - start
    table = rl.to_array(q, world)
    values, policy = rl.state_values(table), rl.greedy_policy(table)
    shortest = 2 * (args.size - 1)
    print(f"{args.size} by {args.size}: {world.n_states} states, {table.size} Q-values, "
          f"{args.episodes} episodes in {taken:.1f}s\n")
    show_policy(world, policy)
    walked = steps_to_goal(E.big_world(args.size, rng=random.Random(args.seed + 2)), q,
                           cap=10 * shortest)
    print(f"\n   S start   G goal   shortest route {shortest} steps")
    print(f"   greedy policy reaches the goal in {walked} steps"
          if walked else "   greedy policy did not reach the goal")
    print(f"   value of the start {values[world.state_index(world.start)]:+.1f}, "
          f"best possible {-shortest:+.1f}")


def run_explore(args):
    world = E.simple_world(rng=random.Random(args.seed))
    q, counts = rl.train_optimistic(world, args.episodes, args.gamma, args.r_plus, args.n_e)
    print(f"the exploring agent, {args.episodes} episodes, "
          f"r_plus {args.r_plus}, n_e {args.n_e}\n")
    print("   policy" + " " * 22 + "times each state was left")
    for row in range(world.height, 0, -1):
        marks, visits = [], []
        for col in range(1, world.width + 1):
            state = (col, row)
            if state in world.walls:
                marks.append("  #  ")
                visits.append("     ")
            elif world.is_terminal(state):
                marks.append(f"{world.terminals[state]:+.0f}".center(5))
                visits.append("    .")
            else:
                marks.append(ARROW[rl.greedy_action(q, state, world.actions)].center(5))
                visits.append(f"{sum(counts.get((state, a), 0) for a in world.actions):5}")
        print("   " + " ".join(marks) + "    " + " ".join(visits))
    least = min((counts.get((s, a), 0), s, a) for s in world.states()
                if not world.is_terminal(s) for a in world.actions)
    print(f"\n   least-tried pair: {least[2]!r} at {least[1]}, tried {least[0]} times")
    print(f"   every pair was tried at least n_e = {args.n_e} times, which is what the "
          f"exploration function is for")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--episodes", type=int, default=3000)
    c = sub.add_parser("cliff")
    c.add_argument("--episodes", type=int, default=500)
    b = sub.add_parser("big")
    b.add_argument("--episodes", type=int, default=5000)
    b.add_argument("--size", type=int, default=30)
    e = sub.add_parser("explore")
    e.add_argument("--episodes", type=int, default=500)
    e.add_argument("--r-plus", type=float, default=2.0)
    e.add_argument("--n-e", type=int, default=5)
    for p in (g, c, b, e):
        p.add_argument("--gamma", type=float, default=1.0)
        p.add_argument("--seed", type=int, default=0)
        if p is not e:
            p.add_argument("--alpha", type=float, default=0.5)
            p.add_argument("--epsilon", type=float, default=0.2 if p is g else 0.1)
    args = parser.parse_args()
    {"grid": run_grid, "cliff": run_cliff, "big": run_big,
     "explore": run_explore}[args.command](args)


if __name__ == "__main__":
    main()
