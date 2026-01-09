class Board:
    def __init__(self):
        self.board = [['','',''],['','',''],['','','']]
    def allowed(self, m):
        if self.board[m[0]][m[1]] == '':
            return True 
        else: 
            return False
    def move(self,m,v):
        
        self.board[m[0]][m[1]] = v
    def draw(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == '':
                    return False
                
        return True 
                
    def win(self,v):
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] == v:
                return True
        for i in range(3):
            if self.board[0][i] == self.board[1][i] == self.board[2][i] == v:
                return True
        if self.board[0][0] == self.board[1][1] == self.board[2][2] == v or self.board[0][2] == self.board[1][1]== self.board[2][0] == v:
            return True
        return False
    def allowed_moves(self):
       moves = []
       for i in range(3):
           for j in range(3):
            if self.board[i][j] == '':
                moves.append((i,j))
       return moves
    
    def print_board(self):
       for i in range(3):
          row = ' | '.join(self.board[i])   
          print(row)
          if i < 2:                          
            print('-' * 9)
    def clone(self):
       new_board = Board()
       new_board.board = [row[:] for row in self.board]
       return new_board


