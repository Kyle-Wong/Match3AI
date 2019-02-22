import tensorflow as tf
import numpy as np
import random

from game import GameState

SEED = 0

class Match3NN:
    def __init___(self,game_state):
        self.game_state = game_state
        self._model = self._make_board_model()
        self._move_array = np.sort(np.array(game_state.get_all_pairs))

    def play_game(turn_count):
    '''
    Play the game for a given number of turns
    '''
        #self.game_state.print_board()
        for i in range(turn_count):
            self.make_move()
        #self.game_state.print_board()

    def make_move():
        '''
        Make a move based on the current state of the game
        '''
        move = self._model.predict([np.array(game_state.board)])
        self.game_state.advance_state(move[0], move[1])
    
    def _make_board_model():
        '''
        Make a new model based on the board size
        '''
        return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(game_state.rows, game_state.cols)),
        tf.keras.layers.Dense(128, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(game_state.rows * (game_state.cols - 1) +
                              game_state.cols * (game_state.rows - 1),
                              activation=tf.nn.softmax)])
    
if __name__ == "__main__":
    random.seed(SEED)
    state = random.getstate()
    gs = GameState(8, 8, 7, state)
    
    nn = Match3NN(gs)
    nn.play_game(10)


