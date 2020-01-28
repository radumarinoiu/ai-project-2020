import json
import time
import numpy as np
import random
import queue
import abc
import matplotlib
from matplotlib import colors, pyplot

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


BOARD_SIZE = 9
RENDER_GAME = False


class State:
    def __init__(self, boardsize, board, active_player, empty_fields, passed, previous_state, fig, image, starting_player=1):
        self.boardsize = boardsize
        self.board = board.copy()
        self.active_player = active_player
        self.empty_fields = empty_fields
        self.passed = passed
        self.previous_state = previous_state
        self.possible_moves = []
        self.finished = False
        self.starting_player = starting_player
        self.image = image
        self.fig = fig

    @property
    def board_repr(self):
        return self.board.reshape((self.boardsize, self.boardsize, 1))

    def get_possible_moves(self):
        self.add_possible_moves()
        return self.possible_moves

    def add_possible_moves(self):
        if len(self.possible_moves) > 0:
            return
        self.add_result_state(True, 0, 0)
        for i in range(self.boardsize):
            for j in range(self.boardsize):
                if self.result_state(False, i, j):
                    self.add_result_state(False, i, j)

    def add_result_state(self, passes, x, y):
        result = self.result_state(passes, x, y)
        if result is not None:
            self.possible_moves.append([passes, x, y, result])

    def result_state(self, passes, x, y):
        if self.board[x,y] != 0 and not passes:
            return None
        new_state = State(self.boardsize, self.board, self.active_player, self.empty_fields, self.passed, self, self.fig, self.image, self.starting_player)
        if new_state.make_move(passes, x, y):
            return new_state
        return None

    def make_move(self, passes, x, y):
        if self.empty_fields < 50 and abs(self.score()) > 50:
            self.finished = True
            self.active_player = -self.active_player
            return True
        if passes:
            if self.passed:
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
        return score + 0.5 * self.starting_player

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
                    print('\N{MULTIPLICATION SIGN IN DOUBLE CIRCLE}', end = '')
                elif field == -1:
                    print('\N{Medium White Circle}', end = '')
            print('\N{Black Square Button}')
        for i in range(self.boardsize + 2):
            print('\N{Black Square Button}', end = '')
        print('')

    def draw_board(self):
        self.image.set_data(self.board)
        self.fig.canvas.draw()
        plt.pause(0.01)

    def clear_possible_moves(self):
        self.possible_moves = []


class Game:
    def __init__(self, agent_one, agent_two, fig, image, boardsize=19):
        starting_player = 1 if np.random.randint(0, 2) else -1
        board = np.zeros((boardsize, boardsize))

        self.state = State(boardsize, board, starting_player, boardsize*boardsize, False, None, fig, image, starting_player)
        self.agent_one = agent_one
        self.agent_two = agent_two
        self.moves_counter = 0

    def winner(self):
        score = self.state.score()
        if score == 0:
            return 0
        elif score > 0:
            return 1
        else:
            return -1

    def moves_count(self):
        return self.moves_counter

    def run_game(self):
        while not self.state.finished:
            result = None
            while result is None:
                if self.state.active_player == 1:
                    passes, x, y = self.agent_one.move(self.state)
                else:
                    passes, x, y = self.agent_two.move(self.state)
                result = self.state.result_state(passes, x, y)
            self.moves_counter += 1
            self.state.clear_possible_moves()
            self.state = result
            # self.state.print_state()
            if RENDER_GAME:
                self.state.draw_board()
            # print(self.state.score())
        winner = self.winner()


class Agent(abc.ABC):
    def __init__(self, name=""):
        self.name = name
        self.epsilon = 0

    def reduce_epsilon(self, d_epsilon):
        self.epsilon = max(0, self.epsilon - d_epsilon)
    def move(self, state):
        pass

    def learn(self):
        pass

    def save(self):
        pass


# game with RandomAgents will  end because they will never pass a turn
class RandomAgent(Agent):
    def move(self, state):
        return not bool(random.randint(0, 50)), random.randrange(state.boardsize), random.randrange(state.boardsize)


class TrainingNeuralNetworkAgent(Agent):
    def __init__(self, name, board_size=9):
        from neuralnetwork import NeuralNetwork
        self.name = name
        self.nn = NeuralNetwork(name=self.name, load=True, learning_rate=0.05, inputs=(board_size, board_size, 1))
        self.epsilon = 1
        self.d_epsilon = 0.9999
        self.do_learn = True

    def learn(self):
        return self.nn.learn()

    def save(self):
        self.nn.save(self.name)

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
        input_board_size = best_move[3].board.shape[0]

        if np.random.rand() >= self.epsilon:
            next_possible_states = np.array([
                move[3].board.reshape((input_board_size, input_board_size, 1)) for move in possible_moves])
            next_scores = self.nn.predict(next_possible_states)
            for index, score in enumerate(next_scores):
                if score > max_score:
                    best_move = possible_moves[index]
                    max_score = score
        best_state: State = best_move[3]

        if self.do_learn:
            self.nn.memorize(best_state.board_repr, (best_state.score()-state.score()) * state.active_player)
        return best_move[0], best_move[1], best_move[2]


class NeuralNetworkAgent(TrainingNeuralNetworkAgent):
    def __init__(self, name, board_size=9):
        super(NeuralNetworkAgent, self).__init__(name, board_size)
        self.learn = False
        self.epsilon = 0


def init_graphics():
    fig = plt.figure()
    ax = fig.gca()
    plt.grid()
    fig.show()
    cmap = colors.ListedColormap(['red', 'white', 'blue'])  # Player2, Neutral, Player1
    image = ax.imshow(np.random.randint(-1, 2, (BOARD_SIZE, BOARD_SIZE)), cmap=cmap, vmin=-1, vmax=1)
    return fig, image


def learn_to_play():
    player1_wins = 0
    player2_wins = 0

    logs = {"mse": [], "win delta": []}

    end_episode = 100
    epsilon_end_episode = 1
    d_epsilon = 1/epsilon_end_episode

    fig, image = init_graphics()

    agent1 = TrainingNeuralNetworkAgent(name="agent1", board_size=BOARD_SIZE)

    # agent2 = TrainingNeuralNetworkAgent(name="agent2", board_size=BOARD_SIZE)
    agent2 = RandomAgent()
    for episode in range(end_episode):
        test = Game(
            agent1, agent2,
            fig, image,
            boardsize=BOARD_SIZE)
        test.run_game()
        agent1_mse = agent1.learn()
        agent1.reduce_epsilon(d_epsilon)
        agent2.learn()
        agent2.reduce_epsilon(d_epsilon)
        if episode % 10 == 0:
            agent1.save()
            agent2.save()
        if test.winner() == 1:
            player1_wins += 1
        if test.winner() == -1:
            player2_wins += 1
        if test.moves_count() < 50:
            time.sleep(1)
        print("Game #{} - Epsilon: {} - Moves: {} - Score: {}:{}".format(
            episode,
            agent1.epsilon,
            test.moves_count(),
            player1_wins, player2_wins))
        logs["mse"].append(float(agent1_mse))
        logs["win delta"].append(player1_wins - player2_wins)

    with open("logs.json", "w") as f:
        f.write(json.dumps(logs))
    pyplot.ylim(
        np.min(np.array(logs["win delta"])),
        np.max(np.array(logs["win delta"]))
    )
    pyplot.plot(logs["mse"])
    pyplot.plot(logs["win delta"])
    pyplot.show()


def just_play():
    player1_wins = 0
    player2_wins = 0
    global RENDER_GAME
    RENDER_GAME = True

    fig, image = init_graphics()

    # agent1 = RandomAgent()
    agent1 = NeuralNetworkAgent(name="agent1", board_size=BOARD_SIZE)
    agent2 = RandomAgent()
    # agent2 = NeuralNetworkAgent(name="agent2", board_size=BOARD_SIZE)

    for game_counter in range(300):
        test = Game(
            agent1, agent2,
            fig, image,
            boardsize=BOARD_SIZE)
        test.run_game()
        if test.winner() == 1:
            player1_wins += 1
        if test.winner() == -1:
            player2_wins += 1
        print("Game #{} - Moves: {} - Score: {}:{}".format(
            game_counter+1,
            test.moves_count(),
            player1_wins, player2_wins))


def show_logs():
    with open("logs.json", "r") as f:
        logs = json.load(f.read())
    pyplot.ylim(
        np.min(np.array(logs["win delta"])),
        np.max(np.array(logs["win delta"]))
    )
    pyplot.plot(logs["mse"])
    pyplot.plot(logs["win delta"])
    pyplot.show()


# TODO: Monte Carlo Tree Search Agent as first step towards something similar to AlphaGo?
if __name__ == '__main__':
    learn_to_play()
    # just_play()
