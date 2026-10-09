"""Tic-tac-toe rules.

A state is a tuple of 9 cells ('X', 'O' or ' '), read row by row:

    0 | 1 | 2
    ---------
    3 | 4 | 5
    ---------
    6 | 7 | 8

X always moves first, so whose turn it is can be read from the board itself.
States are immutable tuples, which makes them usable as dictionary keys for
value tables and Q-tables.
"""

EMPTY = ' '
EMPTY_BOARD = (EMPTY,) * 9

LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
]


def to_move(state):
    """Player whose turn it is."""
    return 'X' if state.count('X') == state.count('O') else 'O'


def other(player):
    return 'O' if player == 'X' else 'X'


def legal_moves(state):
    return [i for i, cell in enumerate(state) if cell == EMPTY]


def play(state, move):
    """Return the state after the player to move places a mark on `move`."""
    if state[move] != EMPTY:
        raise ValueError(f"cell {move} is already taken")
    cells = list(state)
    cells[move] = to_move(state)
    return tuple(cells)


def winner(state):
    """'X' or 'O' if that player has three in a row, otherwise None."""
    for a, b, c in LINES:
        if state[a] != EMPTY and state[a] == state[b] == state[c]:
            return state[a]
    return None


def is_terminal(state):
    return winner(state) is not None or EMPTY not in state


def reward(next_state):
    """Reward for the move that produced `next_state`: +1 if it won, else 0.

    Only the player who just moved can have completed a line, so any winner
    in `next_state` is the mover.
    """
    return 1.0 if winner(next_state) is not None else 0.0


def all_states():
    """Every state reachable from the empty board (5,478 in total)."""
    seen = {EMPTY_BOARD}
    frontier = [EMPTY_BOARD]
    while frontier:
        state = frontier.pop()
        if is_terminal(state):
            continue
        for move in legal_moves(state):
            nxt = play(state, move)
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return seen


def render(state):
    """Board as text, with free cells numbered 1-9 so a human can pick one."""
    cells = [c if c != EMPTY else str(i + 1) for i, c in enumerate(state)]
    rows = [' ' + ' | '.join(cells[r * 3:r * 3 + 3]) for r in range(3)]
    return '\n-----------\n'.join(rows)
