"""Play tic-tac-toe against the computer in the terminal.

    python play.py                     # you are X and move first
    python play.py --as O              # the computer moves first
    python play.py --agent qlearning   # play the self-taught Q-learning agent
"""
import argparse

from tictactoe import board
from tictactoe.agents import QLearningAgent, ValueIterationAgent
from tictactoe.qlearning import QLearning
from tictactoe.solver import ValueIteration


def ask_move(state):
    while True:
        text = input('Your move (1-9): ').strip()
        if text.isdigit() and 1 <= int(text) <= 9 and state[int(text) - 1] == board.EMPTY:
            return int(text) - 1
        print('Pick a free cell between 1 and 9.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--agent', choices=['value-iteration', 'qlearning'], default='value-iteration')
    parser.add_argument('--as', dest='human', choices=['X', 'O'], default='X', help='X moves first')
    parser.add_argument('--episodes', type=int, default=200_000, help='self-play games for Q-learning')
    args = parser.parse_args()

    if args.agent == 'qlearning':
        print(f'Training the Q-learning agent on {args.episodes:,} self-play games (a few seconds)...')
        ai = QLearningAgent(QLearning().train(args.episodes))
    else:
        ai = ValueIterationAgent(ValueIteration().solve())

    state = board.EMPTY_BOARD
    print(f'\nYou are {args.human}. X moves first.\n')
    print(board.render(state))
    while not board.is_terminal(state):
        if board.to_move(state) == args.human:
            move = ask_move(state)
        else:
            move = ai.act(state)
            print(f'Computer plays {move + 1}')
        state = board.play(state, move)
        print()
        print(board.render(state))

    w = board.winner(state)
    print('\nDraw!' if w is None else '\nYou win!' if w == args.human else '\nComputer wins!')


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print('\nBye!')
