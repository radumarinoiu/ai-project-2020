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

class Game:
    def __init__(self, agent_one, agent_two, boardsize=19):
        self.boardsize = boardsize
        self.board = np.zeros((boardsize, boardsize))
        self.agent_one = agent_one
        self.agent_two = agent_two
        self.active_player = 1
        self.empty_fields = boardsize * boardsize
        self.history = [[self.empty_fields, self.board]]
        self.passed = False

    def print_state(self):  # this just blurts emojis into to the console, should this be improved?
        for i in range(self.boardsize + 2):
            print('\N{Black Square Button}', end='')
        print('')
        i, j = 0, 0
        for row in self.board:
            print('\N{Black Square Button}', end='')
            for field in row:
                if field == 0:
                    print('\N{IDEOGRAPHIC SPACE}', end='')
                    board[i][j] = "grey"
                elif field == 1:
                    print('\N{Medium Black Circle}', end='')
                    board[i][j] = "black"
                elif field == -1:
                    print('\N{Medium White Circle}', end='')
                    board[i][j] = "white"
                j += 1
            print('\N{Black Square Button}')
            j = 0
            i += 1
        for i in range(self.boardsize + 2):
            print('\N{Black Square Button}', end='')
        print('')
        print('')

    def updater(self):
        self.draw_board()
        self.after(100, self.updater())

    def winner(self):
        score = self.score()
        if score == 0:
            return 0
        elif score > 0:
            return 1
        else:
            return -1

    def move_is_legal(self, x_coord, y_coord):  # TODO: Ko rule
        return self.board[x_coord, y_coord] == 0

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
        candidates = [(x_coord + 1, y_coord), (x_coord - 1, y_coord), (x_coord, y_coord + 1), (x_coord, y_coord - 1)]
        neighbours = []
        for (x, y) in candidates:
            if x >= 0 and x < self.boardsize and y >= 0 and y < self.boardsize:
                neighbours.append((x, y))
        return neighbours

    def score(self):
        checked = np.zeros((self.boardsize, self.boardsize), dtype=bool)
        score = 0
        for x in range(self.boardsize):
            for y in range(self.boardsize):
                colour, group = self.encircled(checked, (x, y), 0)
                score += colour * len(group) + self.board[x, y]
        return score - 6.5

    def run_game(self):
        while True:
            legal = False
            while not legal:
                if self.active_player == 1:
                    passes, x, y = self.agent_one.move(self)
                else:
                    passes, x, y = self.agent_two.move(self)
                if passes:
                    legal = True
                else:
                    legal = self.move_is_legal(x, y)
            if self.passed and passes:
                return self.winner()
            self.passed = passes
            if not passes:
                self.board[x, y] = self.active_player
                self.empty_fields += self.remove_all_encircled(x, y) - 1
            self.active_player = -self.active_player
            self.history.append([self.empty_fields, self.board])
            self.print_state()
            draw_board(board)
            print(self.score())


class Agent(abc.ABC):
    def move(self, board):
        pass


# game with RandomAgents will never end because they will never pass a turn
class RandomAgent(Agent):
    def move(self, game):
        return False, random.randrange(game.boardsize), random.randrange(game.boardsize)


# TODO: MinMax Agent as first step towards something similar to AlphaGo?

test = Game(RandomAgent(), RandomAgent())
test.run_game()
