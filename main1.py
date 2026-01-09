from board import Board
from player import Player

V = {}

def compute_value(board, c_p, o_p):
    state = tuple(cell for row in board.board for cell in row)
    
    if board.win(c_p.value):
        V[state] = 1
        return 1
    elif board.win(o_p.value):
        V[state] = -1
        return -1
    elif board.draw():
        V[state] = 0
        return 0

    if state in V:
        return V[state]

    maxi = float('-inf')

    for move in board.allowed_moves():
        new_board = board.clone()
        new_board.move(move, c_p.value)
        next_state = tuple(cell for row in new_board.board for cell in row)
        if next_state in V:
            value = -V[next_state]
        else:
            value = -compute_value(new_board, o_p, c_p)
        if value > maxi:
            maxi = value

    V[state] = maxi
    return maxi

# --- Compute all values once ---
b = Board()
player1 = Player('X')
player2 = Player('O')
compute_value(b, player1, player2)

# --- Build P table for all states ---
P = {}
for state in list(V.keys()):
    board = Board()
    board.board = [list(state[i*3:(i+1)*3]) for i in range(3)]
    c_val = 'X' if state.count('X') <= state.count('O') else 'O'

    max_value = float('-inf')
    best_move = None
    for move in board.allowed_moves():
        new_board = board.clone()
        new_board.move(move, c_val)
        next_state = tuple(cell for row in new_board.board for cell in row)
        if next_state in V:
         value = -V[next_state]
        else:
    # compute recursively just in case
         value = -compute_value(new_board, Player('O' if c_val=='X' else 'X'), Player(c_val))

        if value > max_value:
            max_value = value
            best_move = move
    P[state] = best_move

print(P[(('', '', '', '', '', '', '', '', ''))])
print(P[('', '', '', '', 'O', '', '', '', '')])


# --- Play human vs AI ---
bo = Board()
human = Player('O')
ai = Player('X')
current_player = human

while True:
    state = tuple(cell for row in bo.board for cell in row)

    if bo.win(ai.value):
        print("AI wins!")
        break
    elif bo.win(human.value):
        print("Human wins!")
        break
    elif bo.draw():
        print("Draw!")
        break

    if current_player == ai:
        move = P[state]
        bo.move(move, ai.value)
        current_player = human
    else:
        i = int(input("Row: "))
        j = int(input("Col: "))
        if bo.allowed((i, j)):
            bo.move((i, j), human.value)
            current_player = ai
        else:
            print("Invalid move, try again")

    bo.print_board()
