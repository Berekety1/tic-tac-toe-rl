from board import Board
from player import Player
V = {}
def compute_value(board,c_p,o_p):
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
    new_board.move(move,c_p.value)
    next_state =  tuple(cell for row in new_board.board for cell in row)
    if next_state in V:
      value = -V[next_state]
    else:
      value= -compute_value(new_board,o_p,c_p)
    if value > maxi:
      maxi = value
  V[state] = maxi
  return maxi
  

b = Board()

starting_state = tuple(cell for row in b.board for cell in row)
player1 = Player('X')
player2 = Player('O')
current_player = player1
opponent_player = player2
compute_value(b, player1, player2)


P = {}


I = 100

for i in range(I):
  over = False
  b = Board()
  player1 = Player('X')
  player2 = Player('O')
  current_player = player1
  opponent_player = player2
  while not over:
    
    current_player, opponent_player = opponent_player, current_player
    state = tuple(cell for row in b.board for cell in row)
    if b.win(current_player.value):
      V[state] = 1
      over = True
    elif b.draw():
      V[state] = 0
      over = True
    elif b.win(opponent_player.value):
      V[state] = -1
      over = True
    max_value = float('-inf')
    best_move = []
    for move in b.allowed_moves():
       new_board = b.clone()
       new_board.move(move,current_player.value)
       next_state = tuple(cell for row in new_board.board for cell in row)
       if next_state in V:
           value = -V[next_state]
       else:
         value = -compute_value(new_board,opponent_player, current_player)
         
        
    
       if  value > max_value:
         max_value = value
         best_move = move
       P[state] = best_move
    if best_move:
      b.move(best_move,current_player.value)
      
      
    
print(P)