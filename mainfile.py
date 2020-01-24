import numpy as np, random, queue, abc

import tkinter as tk
class MINMAAX: # super mega slow don't recomend to use
    def minmax(self, state):
        moves = state.get_possible_moves()
        score = -10000
        for m in moves:
            new_state = state
            score = new_state.score
            new_state.board[m[1]][m[2]] = 1
            score =  -self.bestoponentmove(new_state) + new_state.score
            if score > best_score:
                best_move = m
                best_score = score
        if best_score < 0:
            return "pass"
        for i in score:
            if score[0] != i:
                return best_score
        return random.moves

    def bestoponentmove(self,state):
        moves = state.get_possible_moves()
        score = -10000
        scores = []
        for m in moves:
            new_state = state
            score = new_state.score
            new_state.board[m[1]][m[2]] = 2
            score = -self.bestsecoundmove(new_state) + new_state.score
            scores.append(score)
            if score > best_score:
                best_score = score

    def bestsecoundmove(self,state):
        moves = state.get_possible_moves()
        score = -10000
        for m in moves:
            new_state = state
            score = new_state.score
            new_state.board[m[1]][m[2]] = 2
            score = new_state.score
            if score > best_score:
                best_score = score
        return best_score
_game()