'''
Laura Ploch, 300176, 01143517@pw.edu.pl
Julia Czmut, 300168, 01143509@pw.edu.pl
Task 3. - Two people deterministic games (tic-tac-toe game using MiniMax algorithm with Alpha-Beta pruning)
'''

class TicTacToe:
    def __init__(self):
        self.initialize_game()

    def initialize_game(self):
        # board initialization
        self.state = [['_','_','_'],
                      ['_','_','_'],
                      ['_','_','_']]

        self.player_turn = 'PLAYER'


    def draw_board(self):
        for i in range(0,3):
            for j in range(0,3):
                print('{}|'.format(self.state[i][j]), end=" ")
            print()
        print()


    # check if the move is valid
    def is_move_valid(self, pos_x, pos_y):
        # if positions are invalid - board is 3x3, so possible position indexes are 0, 1 and 2
        if pos_x < 0 or pos_x > 2 or pos_y < 0 or pos_y > 2:
            return False
        # if position is already taken, mo
        elif self.state[pos_x][pos_y] != '_':
            return False
        # if valid move
        else:
            return True


    # check if the game has finished - if yes, return the winner
    def is_game_finished(self):
        # check for a horizontal win
        for i in range(0,3):
            if self.state[i] == ['X', 'X', 'X']:
                return 'X'
            elif self.state[i] == ['O', 'O', 'O']:
                return 'O'
        
        # check for a vertical win
        for i in range(0,3):
            # if all positions in a column are taken by the same player
            if self.state[0][i] != '_' and self.state[0][i] == self.state[1][i] and self.state[1][i] == self.state[2][i]:
                return self.state[0][i]

        # check for diagonal wins
        # upper left - lower right diagonal
        if self.state[0][0] != '_' and self.state[0][0] == self.state[1][1] and self.state[0][0] == self.state[2][2]:
            return self.state[0][0]

        # lower left - upper right diagonal
        if self.state[0][2] != '_' and self.state[0][2] == self.state[1][1] and self.state[0][2] == self.state[2][0]:
            return self.state[0][2]
        
        # check if the board is full
        for i in range(0,3):
            for j in range(0,3):
                # if there's an empty field, the board is not full, so we continue
                if self.state[i][j] == '_':
                    return 'board_not_full'

        # otherwise it's a tie
        return '='


    def max_alpha_beta(self, alpha, beta):
        # initially maxval is set to be worse than the worst option for max
        maxval = -2
        pos_x = None
        pos_y = None

        # check if game has finished - possible results: 'X', 'O', '_', 'board_not_full'
        result = self.is_game_finished()
        if result == 'X':   # player X won
            return (-1, 0, 0)
        elif result == 'O':   # player O won
            return (1, 0, 0)
        elif result == '=':    # it's a tie
            return (0, 0, 0)
        elif result == 'board_not_full':

            # traverse through the board and evaluate possible moves
            for x in range(0,3):
                for y in range(0,3):
                    if self.state[x][y] == '_':
                        self.state[x][y] = 'O'
                        (val, temp1, temp2) = self.min_alpha_beta(alpha, beta)
                        if val > maxval:
                            maxval = val
                            pos_x = x
                            pos_y = y
                        self.state[x][y] = '_'

                        # alpha - best current option for max (AI)
                        # beta - best current option for min (player)
                        if maxval >= beta:
                            return (maxval, pos_x, pos_y)
                        
                        if maxval > alpha:
                            alpha = maxval

        return (maxval, pos_x, pos_y)
                        

    def min_alpha_beta(self, alpha, beta):
        # initially minval is set to be worse than the worst option for min
        minval = 2
        pos_x = None
        pos_y = None
        
        # check if game has finished - possible results: 'X', 'O', '_', 'board_not_full'
        result = self.is_game_finished()
        if result == 'X':   # player X won
            return (-1, 0, 0)
        elif result == 'O':   # player O won
            return (1, 0, 0)
        elif result == '=':    # it's a tie
            return (0, 0, 0)
        elif result == 'board_not_full':
            # traverse through the board and evaluate possible moves
            for x in range(0,3):
                for y in range(0,3):
                    if self.state[x][y] == '_':
                        self.state[x][y] = 'X'
                        (val, temp1, temp2) = self.max_alpha_beta(alpha, beta)
                        if val < minval:
                            minval = val
                            pos_x = x
                            pos_y = y
                        self.state[x][y] = '_'

                        # alpha - best current option for max (AI)
                        # beta - best current option for min (player)
                        if minval <= alpha:
                            return (minval, pos_x, pos_y)

                        if minval < beta:
                            beta = minval

        return (minval, pos_x, pos_y)
                            
    
    def play(self):
        if input("Do you want to go first? Type y/n: ") == "n":
            self.player_turn = "AI"
        
        while True:
            self.draw_board()
            self.result = self.is_game_finished()
            
            # If the game has been finished:
            if self.result != 'board_not_full':
                if self.result == 'X':
                    print("YOU WON!")
                if self.result == 'O':
                    print("AI won!")
                if self.result == '=':
                    print("It's a tie!")

                self.initialize_game()
                return
            
            if self.player_turn == 'PLAYER':
                while True:
                    
                    while True:
                        try:
                            pos_x = int(input('Choose row (1, 2 or 3): ')) - 1
                            pos_y = int(input('Choose column (1, 2 or 3): ')) - 1
                            break
                        except ValueError:
                            print("That was not a valid number! Try again.")
                        
                    if self.is_move_valid(pos_x, pos_y):
                        self.state[pos_x][pos_y] = 'X'
                        self.player_turn = 'AI'
                        break
                    else:
                        print("The move is invalid, try again.")
            
            elif self.player_turn == 'AI':
                (val, pos_x, pos_y) = self.max_alpha_beta(-10, 10)
                self.state[pos_x][pos_y] = 'O'
                self.player_turn = 'PLAYER'


print("Welcome to TicTacToe game with AI created by Julia Czmut and Laura Ploch!")
print("\tYou are X \n\tAI is O")
ticTacToe = TicTacToe()
choice = True
while choice:
    ticTacToe.play()
    if input("Do you want to play again? Type y/n: ") == 'y':
        choice = True
    else:
        choice = False

