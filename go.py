import numpy as np, tkinter as tk, random, queue, abc, time, math

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

def draw_board(board):
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
    #root.after(20, test.run_game())
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
        self.times_visited = 0
        self.value = None
        self.move_candidates = [np.full((boardsize, boardsize), True), True]
        self.candidates_evaluated = 0
        
    def winner(self):
        score = self.score()
        if score == 0:
            return 0
        elif score > 0:
            return 1
        else:
            return -1
        
    def get_possible_moves(self):
        self.add_possible_moves()
        return self.possible_moves
        
    def add_possible_moves(self):
        if self.move_candidates[1]:
            self.add_result_state(True, 0, 0)
        for i in range(self.boardsize):
            for j in range(self.boardsize):
                if self.move_candidates[0][i, j]:
                    self.add_result_state(False, i, j)
                
    def add_result_state(self, passes, x, y):
        self.candidates_evaluated += 1
        if passes:
            self.move_candidates[1] = False
        else:
            self.move_candidates[0][x, y] = False
        result = self.result_state(passes, x, y)
        if result is not None:
            self.possible_moves.append([passes, x, y, result])
        return result
    
    def result_state(self, passes, x, y):
        if (self.board[x,y] != 0 and not passes) or self.finished:
            return None
        new_state = State(self.boardsize, self.board, self.active_player, self.empty_fields, self.passed, self)
        if new_state.make_move(passes, x, y):
            return new_state
        return None
        
    def make_move(self, passes, x, y):
        if passes:
            if self.passed:
                self.finished = True
        elif self.board[x,y] == 0:
            self.board[x, y] = self.active_player
            self.empty_fields += self.remove_all_encircled(x, y)-1
            if not self.move_is_ko_legal(x, y) or not self.move_is_suicide_legal(x, y):
                return False
        else:
            return False
        self.passed = passes
        self.active_player = -self.active_player
        return True
    
    def move_is_suicide_legal(self, x_coord, y_coord):
        return self.board[x_coord, y_coord] != 0
    
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
        return score # - 0.5 break ties, second mover has disadvantage
    
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
        
    def randomize(self, stone_probability):
        spots = [(x,y) for x in range(self.boardsize) for y in range(self.boardsize)]
        random.shuffle(spots)
        for i, j in spots:
            if  random.random() < stone_probability:
                self.make_move(False, i, j)
        
    

class Game:
    def __init__(self, agent_one, agent_two, boardsize=19, concede_threshold = None):
        self.state = State(boardsize, np.zeros((boardsize, boardsize)), 1, boardsize*boardsize, False, None)
        self.agent_one = agent_one
        self.agent_two = agent_two
        self.concede_threshold = concede_threshold
        self.turn = 0

        
    def run_game(self):
        while not self.state.finished:
            result = None
            passes = True
            x = 0
            y = 0
            while result is None:
                if self.state.active_player == 1:
                    passes, x, y = self.agent_one.move(self.state)
                else:
                    passes, x, y = self.agent_two.move(self.state)
                result = self.state.result_state(passes, x, y)
            self.turn += 1
            print("Player", (-self.state.active_player + 3)/2, "makes move:", passes, x, y)
            self.state.clear_possible_moves()
            self.state = result
            score = self.state.score()
            print("Score:", score)
            self.state.print_state()
            #draw_board(board)
            if self.concede_threshold is not None and self.turn > 10 and self.concede_threshold <= abs(score):
                break
        winner = self.state.winner()
        if winner == 0:
            print("Draw?")
        else:
            print("Player {} won".format(winner))


class Agent(abc.ABC):
    def move(self, state):
        pass


class RandomAgent(Agent):
    def move(self, state):
        return not bool(random.randint(0, 1000)), random.randrange(state.boardsize), random.randrange(state.boardsize)

class MCTSAgent(Agent):
    def __init__(self, name, boardsize=19, processing_time = 100.0, visited_states = 1000, exploration = 0.1):
        from neuralnetwork import NeuralNetwork
        self.name = name
        self.boardsize = boardsize
        self.nn = NeuralNetwork(name=self.name, boardsize = boardsize, load=True)
        self.processing_time = processing_time
        self.visited_states = visited_states
        self.exploration = exploration
        self.reset()
        
    def reset(self):
        self.priority_list = []
        for i in range(self.boardsize):
            for j in range(self.boardsize):
                self.priority_list.append([False, i, j])
        self.priority_list.append([True, 0, 0])
        self.priority_list = np.array(self.priority_list)
        self.values = np.zeros((self.boardsize, self.boardsize))
        self.pass_value = 0

    def save(self):
        self.nn.save(self.name)
        
    def priority_value(self, state, total_visits):
        return state.value + self.exploration * math.sqrt(2 * math.log(total_visits)/state.times_visited)
        
    def move(self, state):
        start_time = time.time()
        state.value = 0.5
        while time.time() - start_time < self.processing_time and state.times_visited < self.visited_states:
            #SELECTION
            current_state = state
            while current_state.candidates_evaluated == self.boardsize*self.boardsize+1 and not current_state.finished:
                max_value = -1000
                max_state = None
                for _,_,_,s in current_state.get_possible_moves():
                    priority = self.priority_value(s, state.times_visited)
                    if priority > max_value:
                        max_value = priority
                        max_state = s
                current_state = max_state
            #EXPANSION: use priority list to determine next move to evaluate, 
            #previous high scoring moves are prefered
            if current_state.finished:
                result = current_state
            else:
                while True:
                    passes, x, y = self.priority_list[current_state.candidates_evaluated]
                    result = current_state.add_result_state(passes, x, y)
                    if result is not None:
                        break
            #SIMULATION: instead of random playouts
            #a network trained to predict the winning probabilites deteremines the values
            if result.finished:
                result.value = (1 + result.winner())/2
            else:
                result.value = self.nn.predict(result)[0]
            if result.active_player == 1:
                    result.value = 1 - result.value
            #BACKPROPAGATION
            current_state = result
            value = result.value
            while current_state != state.previous_state:
                current_state.value = (current_state.value * current_state.times_visited + value) / (current_state.times_visited + 1)
                current_state.times_visited += 1
                current_state = current_state.previous_state
                value = 1 - value
        max_visited = 0
        max_move = None
        for passes, x, y, s in state.get_possible_moves():
            print(passes, x, y, s.times_visited, s.value, s.finished)
            new_value = s.times_visited + s.value
            if passes:
                self.pass_value = new_value
            else:
                self.values[x, y] = new_value
            if new_value > max_visited:
                max_visited = new_value
                max_move = [passes, x, y]
                #for next_passes, next_x, next_y, next_s in s.get_possible_moves():
                    #print("-", next_passes, next_x, next_y, next_s.times_visited, next_s.value, next_s.finished)
        #Update priority_list
        self.priority_list = []
        sorted = np.dstack(np.unravel_index(np.argsort(-self.values, axis = None),(self.boardsize, self.boardsize)))[0]
        for i, j in sorted:
            self.priority_list.append([False, i, j])
        self.priority_list.append([True, 0, 0])
        print("Number of states visited:", state.times_visited)
        print("Likelihood of black winning:", self.nn.predict(state)[0][0])
        return max_move
    
    def learn_from_game(self, end_state):
        value = (end_state.winner() + 1)/2
        checked_state = end_state
        while checked_state is not None:
            self.nn.memorize(checked_state, value)
            checked_state = checked_state.previous_state
        self.nn.learn()

def learn_to_play():
    boardsize = 9
    our_agent = MCTSAgent(name="mcts", boardsize=boardsize,  visited_states = 1000)
    while True:
        training_game = Game(our_agent, our_agent, boardsize=boardsize, concede_threshold = 25)
        training_game.state.randomize(random.uniform(0.1, 0.5))
        training_game.run_game()
        our_agent.learn_from_game(training_game.state)
        our_agent.save()
        our_agent.reset()


def just_play():
    boardsize = 9
    player1_wins = 0
    player2_wins = 0
    mcts = MCTSAgent(name="mcts", boardsize=boardsize,  visited_states = 1000)
    for i in range(100):
        mcts.reset()
        test = Game(
            mcts,
            # NeuralNetworkAgent(name="mcts", board_size=board_size),
            RandomAgent(),
            # RandomAgent(),
            boardsize=boardsize,
            concede_threshold = 25)
        test.run_game()
        if test.state.winner() == -1:
            player1_wins += 1
        if test.state.winner() == 1:
            player2_wins += 1
    print("Player1 wins:", player1_wins)
    print("Player2 wins:", player2_wins)


if __name__ == '__main__':
    learn_to_play()
    #just_play()
