"""Players and helpers for pitting them against each other."""
import random

from . import board


class RandomAgent:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def act(self, state):
        return self.rng.choice(board.legal_moves(state))


class ValueIterationAgent:
    """Plays perfectly using the solved value table. Breaks ties randomly."""

    def __init__(self, solver, seed=None):
        self.solver = solver
        self.rng = random.Random(seed)

    def act(self, state):
        return self.rng.choice(self.solver.best_moves(state))


class QLearningAgent:
    """Plays greedily (no exploration) from a trained Q-table.

    Reads the table without changing it, and breaks ties with its own random
    generator so that evaluating mid-training does not affect the training run.
    """

    def __init__(self, learner, seed=None):
        self.learner = learner
        self.rng = random.Random(seed)

    def act(self, state):
        values = self.learner.Q.get(state) or {m: 0.0 for m in board.legal_moves(state)}
        best = max(values.values())
        return self.rng.choice([m for m, v in values.items() if v == best])


def play_game(x_agent, o_agent):
    """Play one game and return the winner ('X', 'O') or None for a draw."""
    state = board.EMPTY_BOARD
    while not board.is_terminal(state):
        agent = x_agent if board.to_move(state) == 'X' else o_agent
        state = board.play(state, agent.act(state))
    return board.winner(state)


def evaluate(agent, opponent, games=1000):
    """Win/draw/loss rates for `agent`, which plays X in half of the games."""
    results = {'win': 0, 'draw': 0, 'loss': 0}
    for i in range(games):
        agent_side = 'X' if i % 2 == 0 else 'O'
        if agent_side == 'X':
            w = play_game(agent, opponent)
        else:
            w = play_game(opponent, agent)
        if w is None:
            results['draw'] += 1
        elif w == agent_side:
            results['win'] += 1
        else:
            results['loss'] += 1
    return {k: v / games for k, v in results.items()}
