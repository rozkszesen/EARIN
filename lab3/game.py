# EARIN Task 3 - Two people deterministic games
# Jan Jakubiak, Urszula Chmielewska

# set possible values of fields to make playing with different symbols possible
AI='X'
PLAYER='O'
EMPTY=' '

#set starting board to use it for the game
STARTING_BOARD = [
    [EMPTY,EMPTY,EMPTY],
    [EMPTY,EMPTY,EMPTY],
    [EMPTY,EMPTY,EMPTY]
]

#function checking if someone won - returns the player that won or boolean False, if the game has not ended
def checkWin(board):
    for i in range(3):
        # checking rows and columns, if someone won there
        if (board[0][i]==board[1][i]==board[2][i] and board[0][i] != EMPTY):
            return board[0][i]
        if (board[i][0]==board[i][1]==board[i][2] and board[i][0] != EMPTY):
            return board[i][0]
    # checking both diagonals to see if someone won there
    if (board[0][0]==board[1][1]==board[2][2] and board[0][0] != EMPTY):
            return board[0][0]
    if (board[0][2]==board[1][1]==board[2][0] and board[0][2] != EMPTY):
            return board[0][2]
    return False

#function checking if there is a draw - returning boolean True or False
def checkDraw(board):
    possibleWin=False
    for i in range(3):
        # checking rows and columns to see if each of them has both an X and an O on them, therefore making it impossible to win on that line
        if not((board[0][i] == AI or board[1][i] == AI or board[2][i] == AI)and(board[0][i] == PLAYER or board[1][i] == PLAYER or board[2][i] == PLAYER)):
            possibleWin=True
        if not((board[i][0] == AI or board[i][1] == AI or board[i][2] == AI)and(board[i][0] == PLAYER or board[i][1] == PLAYER or board[i][2] == PLAYER)):
            possibleWin=True
    # checking both diagonals accordingly
    if not((board[0][0] == AI or board[1][1] == AI or board[2][2] == AI)and(board[0][0] == PLAYER or board[1][1] == PLAYER or board[2][2] == PLAYER)):
            possibleWin=True
    if not((board[0][2] == AI or board[1][1] == AI or board[2][0] == AI)and(board[0][2] == PLAYER or board[1][1] == PLAYER or board[2][0] == PLAYER)):
            possibleWin=True
    return not possibleWin


# function returning appropriate values based on the state of the board
def evaluation(board):
    if checkWin(board) == AI:
        return 10
    if checkWin(board) == PLAYER:
        return -10
    if checkDraw(board) == True:
        return 0
    return 0


def minmax(board, depth, xTurn, alpha, beta):
    if checkDraw(board) == True:
        return evaluation(board)
    # returning the evaluation value minus the depth, to make the algorithm go for the quickest possible way to win
    if checkWin(board) == AI:
        return evaluation(board)-depth

    if checkWin(board) == PLAYER:
        return evaluation(board)+depth

    if xTurn:
        maximum=-100
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY:
                    # for each EMPTY field, we make a move and recursively check what highest value is possible to be reached
                    board[i][j]=AI
                    value=minmax(board, depth+1, not xTurn, alpha, beta)
                    maximum=max(maximum, value)
                    # we revert the change used to calculate the possible values
                    board[i][j]=EMPTY
                    # we change the alpha value and check if beta <= alpha, to make sure we don't waste time checking scenarios that won't be beneficiary
                    alpha=max(alpha, maximum)
                    if beta <= alpha:
                        break
        return maximum

    if not xTurn:
        minimum=100
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY:
                    board[i][j]=PLAYER
                    value=minmax(board, depth+1, not xTurn, alpha, beta)
                    minimum=min(minimum, value)
                    board[i][j]=EMPTY
                    beta=min(beta, minimum)
                    if beta <= alpha:
                        break
        return minimum


def bestMove(board):
    maximum=-100
    coordinates=(0,0)

    for i in range(3):
        for j in range(3):
            # we check all possible moves using the minmax function
            if board[i][j] == EMPTY:
                board[i][j]=AI
                value=minmax(board, 0, False, -100, 100)
                board[i][j]=EMPTY
                # if the value of a specific scenario is higher than the previous one, we change it and save the coordinates of the field resulting in that scenario
                if(value > maximum):
                    coordinates=(i,j)
                    maximum=value
    return coordinates


# printing the formatted board
def printBoard(board):
    for i in range(3):
        print('[',board[i][0],']','[',board[i][1],']','[',board[i][2],']')

# making the specified move
def makeMove(board, coordinates, sign):
    (i,j)=coordinates
    board[i][j]=sign
    return board


# checking if the move is legal - if the chosen coordinates are between 0 and 2, and if the chosen field is EMPTY
def validMove(board, coordinates):
    (i,j)=coordinates
    if not 0<=i<=2 or not 0<=j<=2:
        print("That's an illegal move - coordinates are not between 1 and 3. Enter a legal set of coordinates")
        return False
    if board[i][j] != EMPTY:
        print("That's an illegal move - that location on the board is already taken. Enter a legal set of coordinates")
        return False
    return True




def game_loop(board):
    # we set the board as the starting board and set the starting player as the player
    board=STARTING_BOARD
    playerTurn=True

    # we ask the player if he wants to go first or second, and change who starts accordingly
    decision=input("Welcome to tic-tac-toe. Input Y if you want to start the game, or N if you want to go second\n")
    while decision!= 'Y' and decision != 'N':
        decision=input("Incorrect character inputted. Input Y if you want to start the game, or N if you want to go second\n")
    if decision == 'N':
        playerTurn=False
    print("You play as '",PLAYER,"', the AI plays as '",AI,"'. Good luck - you're going to need it!")
    printBoard(board)
    # the game runs until either a win or a draw are reached
    while (checkDraw(board)==False and checkWin(board)==False):
        #AI turn
        if not playerTurn:
            print('AI move')
            board=makeMove(board, bestMove(board), AI)
            printBoard(board)
        # Player turn
        if playerTurn:
            print('Your move. Enter coordinates of your move - first row number, then column number. Both row and column have to be between 1 and 3.')
            row=int(input('Enter the row number\n'))
            column=int(input('Enter the column number\n'))
            # checking if the inputted coordinates are valid 
            while not validMove(board, (row-1, column-1)):
                row=int(input('Enter the row number\n'))
                column=int(input('Enter the column number\n'))
            board=makeMove(board, (row-1, column-1), PLAYER)
            printBoard(board)
        # swapping the player's turn
        playerTurn = not playerTurn

    # Printing out the appropriate game result
    if checkWin(board)!=False:
        print("The game ended, the winner is ", checkWin(board))
        print("Final board:")
        printBoard(board)
    if checkDraw(board)==True:
        print("The game ended in a draw")
        print("Final board:")
        printBoard(board)
    return 0

# Running the game
game_loop(STARTING_BOARD)
