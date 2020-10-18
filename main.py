from numpy.core.numeric import indices
import pygame
from pygame import *
import numpy as np
import time
from typing import Tuple
import sys

def record_time(f):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        res = f(*args, **kwargs)
        t2 = time.time()
        print(f"time of ai play: {t2-t1}")
        return res
    return wrapper

GameSize = (700, 700)
BackgroundColor = (213, 184, 154)
Black = (0, 0, 0)
White = (255, 255, 255)
StartPos = 50
Interval = 40
BoardSize = 16
EndPos = StartPos + (BoardSize - 1) * Interval
ChessRadius = 10
NoChess = 0
BlackChess = 1
WhiteChess = 2
PadChess = 3
BlackPlayer = 0
WhitePlayer = 1

class Player:
    def __init__(self, player):
        self.player = player

    def reverse(self):
        self.player = WhitePlayer if self.player == BlackPlayer else BlackPlayer

    @property
    def color(self):
        return White if self.player == WhitePlayer else Black

    @property
    def chess(self):
        return WhiteChess if self.player == WhitePlayer else BlackChess

class Game:
    def __init__(self):
        pygame.init()
        self.surface = pygame.display.set_mode(GameSize, 0, 32)
        self.surface.fill(BackgroundColor)
        self.draw_checkerboard()
        self.state = np.zeros((BoardSize, BoardSize), int)
        self.curr_player = Player(WhitePlayer)
        self.n_policy_to_explore = 7
        self.n_look_forward = 3
        self.weights = {
            'wulianzhu': 1e6,
            'live_4': 1e5,
            'dead_4': 1e4,
            'live_3': 1e3,
            'dead_3': 1e2,
            'live_2': 1e1,
            'dead_2': 1e0,
        }

    def play(self):
        while True:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            event = pygame.event.wait()
            if event.type == QUIT:
                pygame.quit()
            elif event.type == MOUSEBUTTONDOWN:
                if event.button != 1:
                    continue
                if self.in_checkerboard(mouse_x, mouse_y):
                    y, x = self.checkboard_pos(mouse_x, mouse_y)
                    if self.state[x, y] != NoChess:
                        continue
                    self.draw_chess(
                        checkerboard_x = y,
                        checkerboard_y = x,
                        color = self.curr_player.color,
                    )
                    self.state[x, y] = self.curr_player.chess
                    self.curr_player.reverse()
                    ai_x, ai_y = self.policy()
                    self.draw_chess(
                        checkerboard_x = ai_y,
                        checkerboard_y = ai_x,
                        color = self.curr_player.color,
                    )
                    self.state[ai_x, ai_y] = self.curr_player.chess
                    self.curr_player.reverse()

    @record_time
    def policy(self) -> Tuple[int, int]:
        value, policy = self.alpha_beta_prune(-sys.maxsize, sys.maxsize, self.n_look_forward)
        return policy

    def alpha_beta_prune(self, alpha, beta, depth) -> Tuple[int, Tuple[int, int]]:
        policys = []
        scores = []
        for x, y in self.valid_pos():
            self.state[x, y] = self.curr_player.chess
            self.curr_player.reverse()
            scores.append(self.evaluate())
            policys.append((x, y))
            self.curr_player.reverse()
            self.state[x, y] = NoChess
        indices = [i for i, x in sorted(enumerate(scores), key=lambda x: x[1])]
        policys = [policys[i] for i in indices]
        scores = [scores[i] for i in indices]

        if depth == 1:
            if self.curr_player.player == WhitePlayer:
                return scores[-1], policys[-1]
            else:
                return scores[0], policys[0]
        else:
            policy = (None, None)
            if self.curr_player.player == WhitePlayer:
                value = -sys.maxsize
                for (x, y) in policys[::-1][:self.n_policy_to_explore]:
                    self.state[x, y] = self.curr_player.chess
                    self.curr_player.reverse()
                    child_value, _ = self.alpha_beta_prune(alpha, beta, depth-1)
                    self.curr_player.reverse()
                    self.state[x, y] = NoChess
                    if child_value > value:
                        value = child_value
                        policy = (x, y)
                        alpha = max(alpha, value)
                        if value >= beta:
                            break
                return value, policy
            else:
                value = sys.maxsize
                for (x, y) in policys[:self.n_policy_to_explore]:
                    self.state[x, y] = self.curr_player.chess
                    self.curr_player.reverse()
                    child_value, _ = self.alpha_beta_prune(alpha, beta, depth-1)
                    self.curr_player.reverse()
                    self.state[x, y] = NoChess
                    if child_value < value:
                        value = child_value
                        policy = (x, y)
                        beta = min(beta, value)
                        if value <= alpha:
                            break
                return value, policy

    def evaluate(self) -> int:
        white_score = 0
        black_score = 0
        white_n_wulianzhu, black_n_wulianzhu = self.n_wulianzhu()
        white_score += self.weights['wulianzhu'] * white_n_wulianzhu
        black_score += self.weights['wulianzhu'] * black_n_wulianzhu
        for i in range(3, 5):
            white_n_live, white_n_dead, black_n_live, black_n_dead = self.n_live_dead(i)
            white_score += self.weights[f'live_{i}'] * white_n_live
            white_score += self.weights[f'dead_{i}'] * white_n_dead
            black_score += self.weights[f'live_{i}'] * black_n_live
            black_score += self.weights[f'dead_{i}'] * black_n_dead
        if self.curr_player.player == WhitePlayer:
            return white_score * 10 - black_score
        else:
            return white_score - black_score * 10

    def n_wulianzhu(self) -> int:
        fragments = []
        a, b = np.mgrid[0:BoardSize-4, 0:5]
        index = a + b
        fragments.append(self.state[:, index].reshape(-1, 5))
        fragments.append(self.state.T[:, index].reshape(-1, 5))
        fragments.append(self.state[np.expand_dims(index, 1), index].reshape(-1, 5))
        fragments.append(np.fliplr(self.state)[np.expand_dims(index, 1), index].reshape(-1, 5))
        fragments = np.concatenate(fragments, axis=0)
        res = []
        for for_chess in (WhiteChess, BlackChess):
            res.append(np.all(fragments==for_chess, axis=1).sum())
        return res

    def n_live_dead(self, n_chess) -> int:
        state = np.pad(self.state, pad_width=1, mode='constant', constant_values=PadChess)
        fragments = []
        a, b = np.mgrid[0:BoardSize-(n_chess+1), 0:n_chess+2]
        index = a + b
        fragments.append(state[1:-1, index].reshape(-1, n_chess+2))
        fragments.append(state.T[1:-1, index].reshape(-1, n_chess+2))
        fragments.append(state[np.expand_dims(index, 1), index].reshape(-1, n_chess+2))
        fragments.append(np.fliplr(state)[np.expand_dims(index, 1), index].reshape(-1, n_chess+2))
        fragments = np.concatenate(fragments, axis=0)
        res = []
        for for_chess in (WhiteChess, BlackChess):
            res.append((np.all(fragments[:, 1:-1]==for_chess, axis=1) & (fragments[:, 0]==NoChess) & (fragments[:, -1]==NoChess)).sum())
            res.append((np.all(fragments[:, 1:-1]==for_chess, axis=1) & ((fragments[:, 0]==NoChess) | (fragments[:, -1]==NoChess)) & \
                    (fragments[:, 0]!=for_chess) & (fragments[:, -1]!=for_chess) & (fragments[:, 0]!=fragments[:, -1])).sum())
        return res

    def valid_pos(self):
        state = np.pad(self.state, pad_width=1, mode='constant', constant_values=NoChess)
        index = np.array([np.arange(0, BoardSize), np.arange(1, BoardSize+1), np.arange(2, BoardSize+2)])
        left = state[index, :-2]
        right = state[index, 2:]
        top = state[:-2, index]
        bottom = state[2:, index]
        valid_index = np.nonzero(((left != NoChess).any(axis=0) | (right != NoChess).any(axis=0) | \
            (top != NoChess).any(axis=1) | (bottom != NoChess).any(axis=1)) & (state[1:-1, 1:-1] == NoChess))
        return [(x, y) for x, y in zip(valid_index[0], valid_index[1])]

    def checkboard_pos(self, mouse_x, mouse_y):
        x = round(abs(mouse_x - StartPos) / Interval)
        y = round(abs(mouse_y - StartPos) / Interval)
        return x, y

    def surface_pos(self, checkerboard_x, checkerboard_y):
        x = checkerboard_x * Interval + StartPos
        y = checkerboard_y * Interval + StartPos
        return x, y

    def in_checkerboard(self, mouse_x, mouse_y):
        return mouse_x >= StartPos - Interval/2 and mouse_x <= EndPos + Interval/2 \
            and mouse_y >= StartPos - Interval/2 and mouse_y <= EndPos + Interval/2

    def draw_chess(self, checkerboard_x, checkerboard_y, color):
        pygame.draw.circle(self.surface, color, self.surface_pos(checkerboard_x, checkerboard_y), ChessRadius)
        pygame.display.update()

    def draw_checkerboard(self):
        for x in range(StartPos, EndPos+1, Interval):
            pygame.draw.line(self.surface, Black, (x, StartPos), (x, EndPos), 2)
            pygame.draw.line(self.surface, Black, (StartPos, x), (EndPos, x), 2)
        pygame.display.update()

game = Game()
game.play()
