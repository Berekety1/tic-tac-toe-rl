"""Train the Q-learning agent by self-play and compare it with value iteration.

    python train.py                    # 200k games, then saves the learning curve
    python train.py --episodes 50000

During training the agent is regularly tested on three things:
  * policy agreement: in how many of the 4,520 non-terminal positions its
    greedy move is one of the optimal moves found by value iteration
  * loss rate against a random player
  * loss rate against the perfect (value iteration) player
"""
import argparse
import time
from pathlib import Path

from tictactoe import board
from tictactoe.agents import QLearningAgent, RandomAgent, ValueIterationAgent, evaluate
from tictactoe.qlearning import QLearning
from tictactoe.solver import ValueIteration

ASSETS = Path(__file__).parent / 'assets'


def policy_agreement(learner, optimal):
    """Fraction of positions where every greedy Q move is an optimal move."""
    agree = 0
    for state, best in optimal.items():
        q = learner.Q.get(state)
        if not q:
            continue
        top = max(q.values())
        if {m for m, v in q.items() if v == top} <= best:
            agree += 1
    return agree / len(optimal)


def plot(history, path, dark=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if dark:
        surface, ink, muted, grid = '#1a1a19', '#ffffff', '#c3c2b7', '#383835'
        colors = ['#3987e5', '#d95926', '#199e70']
    else:
        surface, ink, muted, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
        colors = ['#2a78d6', '#eb6834', '#1baf7a']

    episodes = [h['episode'] / 1000 for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150, facecolor=surface)
    panels = [
        ('Agreement with the optimal policy', [
            ('agreement', 'Optimal move chosen', colors[0]),
        ]),
        ('Games lost by the Q-learning agent', [
            ('loss_random', 'vs random player', colors[1]),
            ('loss_perfect', 'vs perfect player', colors[2]),
        ]),
    ]
    for ax, (title, series) in zip(axes, panels):
        ax.set_facecolor(surface)
        labelled = set()
        for key, label, color in series:
            values = [100 * h[key] for h in history]
            ax.plot(episodes, values, color=color, linewidth=2, label=label)
            end_label = f'{values[-1]:.0f}%'
            if end_label in labelled:  # lines ending on the same value share one label
                continue
            labelled.add(end_label)
            ax.annotate(end_label, (episodes[-1], values[-1]),
                        xytext=(6, 0), textcoords='offset points',
                        va='center', fontsize=9, color=ink)
        ax.set_title(title, loc='left', fontsize=12, color=ink, pad=10)
        ax.set_xlabel('Self-play games (thousands)', fontsize=9, color=muted)
        ax.set_ylim(-3, 103)
        ax.yaxis.set_major_formatter(lambda v, _: f'{v:.0f}%')
        ax.grid(axis='y', color=grid, linewidth=0.8)
        ax.tick_params(colors=muted, labelsize=8, length=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(grid)
        if len(series) > 1:
            ax.legend(frameon=False, fontsize=9, labelcolor=ink, loc='upper right')
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episodes', type=int, default=200_000)
    parser.add_argument('--eval-every', type=int, default=2_000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()

    start = time.time()
    solver = ValueIteration().solve()
    print(f'Value iteration: {len(solver.states):,} states, converged in {solver.sweeps} sweeps '
          f'({time.time() - start:.2f}s). Value of the empty board: {solver.V[board.EMPTY_BOARD]:+.1f}')
    optimal = {s: set(solver.best_moves(s)) for s in solver.states if not board.is_terminal(s)}

    learner = QLearning(seed=args.seed)
    random_player = RandomAgent(seed=1)
    perfect_player = ValueIterationAgent(solver, seed=2)
    history = []

    def checkpoint(episode):
        agent = QLearningAgent(learner, seed=3)
        history.append({
            'episode': episode,
            'agreement': policy_agreement(learner, optimal),
            'loss_random': evaluate(agent, random_player, 500)['loss'],
            'loss_perfect': evaluate(agent, perfect_player, 200)['loss'],
        })
        h = history[-1]
        if episode % 10_000 == 0:
            print(f'  {episode:>7,} games | agreement {h["agreement"]:6.1%} | '
                  f'loss vs random {h["loss_random"]:5.1%} | loss vs perfect {h["loss_perfect"]:5.1%}')

    print(f'Training Q-learning for {args.episodes:,} self-play games...')
    start = time.time()
    learner.train(args.episodes, callback=checkpoint, every=args.eval_every)
    print(f'Done in {time.time() - start:.1f}s (including evaluation). Q-table covers {len(learner.Q):,} states.')

    agent = QLearningAgent(learner, seed=3)
    print('\nFinal results over 2,000 games each (agent plays X in half of them):')
    print('| Agent | Opponent | Win | Draw | Loss |')
    print('|---|---|---|---|---|')
    for name, player in [('Value iteration', ValueIterationAgent(solver, seed=4)), ('Q-learning', agent)]:
        for opp_name, opp in [('Random', RandomAgent(seed=5)), ('Value iteration', ValueIterationAgent(solver, seed=6))]:
            r = evaluate(player, opp, 2000)
            print(f'| {name} | {opp_name} | {r["win"]:.1%} | {r["draw"]:.1%} | {r["loss"]:.1%} |')

    if not args.no_plot:
        plot(history, ASSETS / 'learning_curve.png')
        plot(history, ASSETS / 'learning_curve_dark.png', dark=True)
        print(f'\nSaved {ASSETS / "learning_curve.png"} (and a dark-mode version)')


if __name__ == '__main__':
    main()
