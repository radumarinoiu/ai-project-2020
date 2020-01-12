import numpy as np, random, queue, abc

import tkinter as tk

board = [[None for j in range(19)] for i in range(19)]  # separate board for gui

def draw_white(i, j, event):
    event.widget.config(bg="white")
    board[i][j] = "white"

def draw_black(i, j, event):
    event.widget.config(bg="white")
    board[i][j] = "white"

def draw_grey(i, j, event):
    event.widget.config(bg="grey")
    board[i][j] = "grey"

def draw_board(board): # TODO: Make it run in the same window and clean up code
    root = tk.Tk()
    root.title("Go")
    root.geometry("345x400")
    root.configure(background='grey')
    for i, row in enumerate(board):
        for j, column in enumerate(row):
            if board[i][j] == "grey":
                # L = tk.Label(root, text='    ', bg='grey')
                # L.grid(row=i, column=j)
                # L.bind('<Button-1>', lambda e, i=i, j=j: draw_grey(i, j, e))
                pass
            elif board[i][j] == "black":
                L = tk.Label(root, text='    ', bg='black')
                L.grid(row=i, column=j)
                L.bind('<Button-1>', lambda e, i=i, j=j: draw_black(i, j, e))
            elif board[i][j] == "white":
                L = tk.Label(root, text='    ', bg='white')
                L.grid(row=i, column=j)
                L.bind('<Button-1>', lambda e, i=i, j=j: draw_white(i, j, e))

    root.update()
    root.after(20, test.run_game())
    root.after(10, draw_board(board))
    root.mainloop()

class State:
    def __init__(self, boardsize, board, active_player, empty_fields, passed, previous_state):
        self.boardsize = boardsize
        self.board = board.copy()
        self.active_player = active_player
        self.empty_fields = empty_fields
        self.passed = passed
        self.previous_state = previous_state
        self.possible_moves = []
        self.finished = False
        
    def get_possible_moves(self):
        self.add_possible_moves()
        return self.possible_moves
        
    def add_possible_moves(self):
        if len(self.possible_moves) > 0:
            return
        self.add_result_state(True, 0, 0)
        for i in range(self.boardsize):
            for j in range(self.boardsize):
                self.add_result_state(False, i, j)
                
    def add_result_state(self, passes, x, y):
        result = self.result_state(passes, x, y)
        if result is not None:
            self.possible_moves.append([passes, x, y, result])
    
    def result_state(self, passes, x, y):
        if self.board[x,y] != 0:
            return None
        new_state = State(self.boardsize, self.board, self.active_player, self.empty_fields, self.passed, self)
        if new_state.make_move(passes, x, y):
            return new_state
        return None
        
    def make_move(self, passes, x, y):
        if passes and self.passed:
            self.finished = True
        elif self.board[x,y] == 0:
            self.board[x, y] = self.active_player
            self.empty_fields += self.remove_all_encircled(x, y)-1
            if not self.move_is_ko_legal(x, y):
                return False
        else:
            return False
        self.passed = passes
        self.active_player = -self.active_player
        return True
    
    def move_is_ko_legal(self, x_coord, y_coord):
        checked_state = self.previous_state
        while checked_state is not None:
            if checked_state.empty_fields == self.empty_fields:
                if self.board_is_equal(checked_state.board):
                    return False
            checked_state = checked_state.previous_state
        return True
    
    def board_is_equal(self, test_board):
        for i in range(self.boardsize):
            for j in range(self.boardsize):
                if self.board[i,j] != test_board[i,j]:
                    return False
        return True
    
    def remove_all_encircled(self, x_coord, y_coord):
        colour = self.board[x_coord, y_coord]
        checked = np.zeros((self.boardsize, self.boardsize), dtype=bool)
        removed = 0
        for n in self.neighbours(x_coord, y_coord):
            removed += self.remove_encircled(checked, n, -colour)
        removed += self.remove_encircled(checked, (x_coord, y_coord), colour)
        return removed
    
    def remove_encircled(self, checked, start, colour):
        removed = 0
        for (x, y) in self.encircled_by(checked, start, colour, -colour):
            self.board[x, y] = 0
            removed += 1
        return removed
    
    def encircled_by(self, checked, start, colour, encircling_colour):
        found_colour, group = self.encircled(checked, start, colour)
        if found_colour == encircling_colour:
            return group
        else:
            return []
    
    def encircled(self, checked, start, colour):
        q = queue.Queue()
        q.put(start)
        current_group = []
        neighbour_colour_found = False
        encircling_colour = 0
        encircled = True
        while not q.empty():
            element_x, element_y = q.get()
            if self.board[element_x, element_y] == colour:
                if not checked[element_x, element_y]:
                    current_group.append((element_x, element_y))
                    for neighbour in self.neighbours(element_x, element_y):
                        q.put(neighbour)
                    checked[element_x, element_y] = True
            elif not neighbour_colour_found:
                encircling_colour = self.board[element_x, element_y]
                neighbour_colour_found = True
            elif not self.board[element_x, element_y] == encircling_colour:
                encircled = False
        if encircled:
            return encircling_colour, current_group
        else:
            return encircling_colour, []        
                
    def neighbours(self, x_coord, y_coord):
        candidates = [(x_coord+1, y_coord),(x_coord-1, y_coord),(x_coord, y_coord+1),(x_coord, y_coord-1)]
        neighbours = []
        for (x, y) in candidates:
            if x>=0 and x < self.boardsize and y>=0 and y < self.boardsize:
                neighbours.append((x, y))
        return neighbours
    
    def score(self):
        checked = np.zeros((self.boardsize, self.boardsize), dtype=bool)
        score = 0
        for x in range(self.boardsize):
            for y in range(self.boardsize):
                colour, group = self.encircled(checked, (x, y), 0)
                score += colour*len(group)+self.board[x,y]
        return score - 6.5  
    
    def print_state(self):  # this just blurts emojis into to the console, should this be improved?
        for i in range(self.boardsize + 2):
            print('\N{Black Square Button}', end = '')
        print('')
        for row in self.board:
            print('\N{Black Square Button}', end = '')
            for field in row:
                if field == 0:
                    print('\N{IDEOGRAPHIC SPACE}', end = '')
                elif field == 1:
                    print('\N{Medium Black Circle}', end = '')
                elif field == -1:
                    print('\N{Medium White Circle}', end = '')
            print('\N{Black Square Button}')
        for i in range(self.boardsize + 2):
            print('\N{Black Square Button}', end = '')
        print('')
        print('')
        
    def clear_possible_moves(self):
        self.possible_moves = []
    

class Game:
    def __init__(self, agent_one, agent_two, boardsize = 19):
        self.state = State(boardsize, np.zeros((boardsize, boardsize)), 1, boardsize*boardsize, False, None)
        self.agent_one = agent_one
        self.agent_two = agent_two
    
    def winner(self):
        score = self.state.score()
        if score == 0:
            return 0
        elif score > 0:
            return 1
        else:
            return -1
        
    def run_game(self):
        while not self.state.finished:
            result = None
            while result is None:
                if self.state.active_player == 1:
                    passes, x, y = self.agent_one.move(self.state)
                else:
                    passes, x, y = self.agent_two.move(self.state)
                result = self.state.result_state(passes, x, y)
            self.state.clear_possible_moves()
            self.state = result
            self.state.print_state()
            #draw_board(board)
            print(self.state.score())
        print(self.winner())


class Agent(abc.ABC):
    def move(self, state):
        pass


# game with RandomAgents will  end because they will never pass a turn
class RandomAgent(Agent):
    def move(self, state):
        return False, random.randrange(state.boardsize), random.randrange(state.boardsize)


class NeuralNetworkAgent(Agent):
    def __init__(self):
        from neuralnetwork import NeuralNetwork
        self.nn = NeuralNetwork(load=True)

    def move(self, state):
        best_move = None
        max_score = 0
        for move in state.get_possible_moves():
            score = self.nn.predict(move[3])[0]
            if score > max_score:
                max_score = score
                best_move = move
        return best_move[0], best_move[1], best_move[2]


class TrainingNeuralNetworkAgent(Agent):
    def __init__(self):
        from neuralnetwork import NeuralNetwork
        self.nn = NeuralNetwork(load=False, learning_rate=0.5, inputs=9*9)
        self.epsilon = 1
        self.d_epsilon = 0.9999

    def move(self, state):
        possible_moves = state.get_possible_moves()
        try:
            best_move = random.choice(list(possible_moves))
        except IndexError:
            print(len(list(possible_moves)))
            print(possible_moves)
            raise
        max_score = 0
        self.epsilon *= self.d_epsilon
        if np.random.rand() >= self.epsilon:
            for move in possible_moves:
                score = self.nn.predict(move[3])[0]
                self.nn.memorize(move[3], move[3].score())
                self.nn.learn()
                if score > max_score:
                    max_score = score
                    best_move = move
        else:
            self.nn.memorize(best_move[3], best_move[3].score())
        return best_move[0], best_move[1], best_move[2]


# TODO: Monte Carlo Tree Search Agent as first step towards something similar to AlphaGo?
if __name__ == '__main__':
    test = Game(TrainingNeuralNetworkAgent(), RandomAgent(), boardsize=9)
    test.run_game()
