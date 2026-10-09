import pytest

from tictactoe import board
from tictactoe.agents import QLearningAgent, RandomAgent, ValueIterationAgent, evaluate
from tictactoe.qlearning import QLearning
from tictactoe.solver import ValueIteration


def make(rows):
    """Build a state from a picture like 'XO./.X./...' ('.' = empty)."""
    return tuple(board.EMPTY if c == '.' else c for c in rows.replace('/', ''))


@pytest.fixture(scope='module')
def solver():
    return ValueIteration().solve()


def test_winner_detects_rows_columns_and_diagonals():
    assert board.winner(make('XXX/OO./...')) == 'X'
    assert board.winner(make('XO./XO./.O.')) == 'O'
    assert board.winner(make('X.O/.XO/..X')) == 'X'
    assert board.winner(make('XOX/XOO/OXX')) is None


def test_turns_alternate_starting_with_x():
    assert board.to_move(board.EMPTY_BOARD) == 'X'
    assert board.to_move(board.play(board.EMPTY_BOARD, 4)) == 'O'


def test_cannot_play_on_a_taken_cell():
    with pytest.raises(ValueError):
        board.play(make('X../.../...'), 0)


def test_number_of_reachable_states():
    assert len(board.all_states()) == 5478


def test_value_iteration_converges(solver):
    assert solver.deltas[-1] == 0.0
    assert solver.sweeps <= 10


def test_perfect_play_is_a_draw(solver):
    assert solver.V[board.EMPTY_BOARD] == 0.0


def test_takes_a_winning_move(solver):
    # X to move and can win at cell 2
    assert solver.best_moves(make('XX./OO./...')) == [2]


def test_blocks_the_opponent(solver):
    # O to move must block X at cell 2
    assert solver.best_moves(make('XX./O../...')) == [2]


def test_value_iteration_never_loses(solver):
    perfect = ValueIterationAgent(solver, seed=0)
    assert evaluate(perfect, RandomAgent(seed=1), 1000)['loss'] == 0
    assert evaluate(perfect, ValueIterationAgent(solver, seed=2), 200)['draw'] == 1.0


def test_q_learning_learns_to_never_lose(solver):
    agent = QLearningAgent(QLearning(seed=0).train(50_000), seed=0)
    assert evaluate(agent, ValueIterationAgent(solver, seed=1), 200)['loss'] == 0
    assert evaluate(agent, RandomAgent(seed=2), 1000)['loss'] == 0
