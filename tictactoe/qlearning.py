"""Learn tic-tac-toe from scratch with tabular Q-learning and self-play.

One Q-table is shared by both sides: Q(s, a) is the value of playing a in
state s for whoever is to move in s. Because the game is zero-sum, the best
the opponent can get from the next state is our loss, so the update is

    target  = R(s, a) - gamma * max_a' Q(s', a')      (0 if s' is terminal)
    Q(s, a) = Q(s, a) + alpha * (target - Q(s, a))

Unlike value iteration this never looks at the rules ahead of time. It only
learns from the games it plays against itself, exploring with an
epsilon-greedy policy whose epsilon decays over training.
"""
import random

from . import board


class QLearning:
    def __init__(self, alpha=0.5, gamma=0.9, epsilon_start=1.0, epsilon_end=0.05, seed=0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon = epsilon_start
        self.rng = random.Random(seed)
        self.Q = {}  # state -> {move: value}

    def values(self, state):
        if state not in self.Q:
            self.Q[state] = {m: 0.0 for m in board.legal_moves(state)}
        return self.Q[state]

    def greedy(self, state):
        values = self.values(state)
        best = max(values.values())
        return self.rng.choice([m for m, v in values.items() if v == best])

    def choose(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.choice(board.legal_moves(state))
        return self.greedy(state)

    def update(self, state, move, next_state):
        target = board.reward(next_state)
        if not board.is_terminal(next_state):
            target -= self.gamma * max(self.values(next_state).values())
        q = self.values(state)
        q[move] += self.alpha * (target - q[move])

    def play_episode(self):
        state = board.EMPTY_BOARD
        while not board.is_terminal(state):
            move = self.choose(state)
            next_state = board.play(state, move)
            self.update(state, move, next_state)
            state = next_state

    def train(self, episodes, callback=None, every=1000):
        """Self-play for `episodes` games. Epsilon decays linearly over the
        first 80% of training and then stays at epsilon_end.

        If given, callback(episode) is called every `every` episodes.
        """
        decay_episodes = 0.8 * episodes
        for episode in range(1, episodes + 1):
            progress = min(1.0, episode / decay_episodes)
            self.epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
            self.play_episode()
            if callback and episode % every == 0:
                callback(episode)
        return self
