"""Solve tic-tac-toe exactly with value iteration.

Tic-tac-toe is a two-player zero-sum game, so it can be written as an MDP
whose values are always seen from the side of the player to move. After we
move it is the opponent's turn, and their value is the negative of ours.
The Bellman optimality equation then becomes

    V(s) = max_a [ R(s, a) - gamma * V(s') ]      s' = state after move a
    V(s) = 0                                      if s is terminal

where R(s, a) = +1 if move a wins and 0 otherwise. Value iteration starts
from V = 0 and applies this update to every state until nothing changes.
A discount gamma < 1 makes the agent prefer quick wins and slow losses.
"""
from . import board


class ValueIteration:
    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self.states = sorted(board.all_states())
        self.V = {s: 0.0 for s in self.states}
        self.deltas = []  # largest change in V after each sweep

    @property
    def sweeps(self):
        return len(self.deltas)

    def q_value(self, state, move):
        nxt = board.play(state, move)
        return board.reward(nxt) - self.gamma * self.V[nxt]

    def solve(self, tol=1e-12, max_sweeps=100):
        for _ in range(max_sweeps):
            # Synchronous update: every new value is computed from the old table.
            new_V = {}
            for s in self.states:
                if board.is_terminal(s):
                    new_V[s] = 0.0
                else:
                    new_V[s] = max(self.q_value(s, m) for m in board.legal_moves(s))
            delta = max(abs(new_V[s] - self.V[s]) for s in self.states)
            self.V = new_V
            self.deltas.append(delta)
            if delta <= tol:
                break
        return self

    def best_moves(self, state):
        """All moves that reach the optimal value (there are often ties)."""
        values = {m: self.q_value(state, m) for m in board.legal_moves(state)}
        best = max(values.values())
        return [m for m, v in values.items() if best - v < 1e-9]
