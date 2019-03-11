import tensorflow as tf
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt
import game
from game import GameState	

BATCH_SIZE = 100
MAX_EPSILON = 1
MIN_EPSILON = 0
LAMBDA = 0.0002
GAMMA = 1
SEED = 7

class Model:
    def __init__(self, num_states, num_actions, batch_size):
        self.num_states = num_states
        self.num_actions = num_actions
        self.batch_size = batch_size
        # define the placeholders
        self._states = None
        self._actions = None
        # the output operations
        self._logits = None
        self._optimizer = None
        self.var_init = None
        # now setup the model
        self._define_model()

    def _define_model(self):
        self._states = tf.placeholder(shape=[None, self.num_states], dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self.num_actions], dtype=tf.float32)
        # create a couple of fully connected hidden layers
        fc1 = tf.layers.dense(self._states, 100)
        fc2 = tf.layers.dense(fc1, 100)
        self._logits = tf.layers.dense(fc2, self.num_actions)
        loss = tf.losses.log_loss(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self.var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states:
                                                 state.reshape(1, self.num_states)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, x_batch, y_batch):
        sess.run(self._optimizer, feed_dict={self._states: x_batch, self._q_s_a: y_batch})

class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)

class GameRunner:
    def __init__(self, sess, model, env, memory, max_eps, min_eps, decay):
        self._sess = sess
        self._env = env
        self._model = model
        self._memory = memory
        self._max_eps = max_eps
        self._min_eps = min_eps
        self._decay = decay
        self._eps = self._max_eps
        self._steps = 0
        self.reward_store = []
        self._moves = game.get_all_pairs(self._env.rows, self._env.cols)
        self._action_queue = []

    def run(self, render):
        self._env.reset()
        state = np.reshape(self._env.freq_board(), (1,-1))[0]

        if not self._env.calculate_if_moves_left():
            return
        
        tot_reward = 0
        self._action_queue = []
        
        while True:
            action = self._choose_action(state)
            self._action_queue.append(self._moves[action])
            if render:
                self._env.print_board()
                print(self._moves[action])
                    
            next_state, reward, done = self._env.advance_state(self._moves[action][0], self._moves[action][1])
            next_state = np.reshape(self._env.freq_board(), (1,-1))[0]

            self._memory.add_sample((state, action, reward, next_state))
            self._replay()

            # exponentially decay the eps value
            self._steps += 1
            self._eps = self._min_eps + (self._max_eps - self._min_eps) \
                                      * math.exp(-self._decay * self._steps)

            # move the agent to the next state and accumulate the reward
            state = next_state
            tot_reward += reward

            # if the game is done, break the loop
            if done:
                self.reward_store.append(tot_reward)
                break

        print("Step {}, Total reward: {}, Eps: {}".format(self._steps, tot_reward, self._eps))

    def _choose_action(self, state):
        action_list = self._env.get_valid_moves()
        if random.random() < self._eps:
            return action_list[random.randint(0, len(action_list) - 1)]
        else:
            return action_list[np.argmax(self._model.predict_one(state, self._sess)[0][action_list])]

    def _replay(self):
        batch = self._memory.sample(self._model.batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self._model.num_states)
                                 if val[3] is None else val[3]) for val in batch])
        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)
        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)
        # setup training arrays
        x = np.zeros((len(batch), self._model.num_states))
        y = np.zeros((len(batch), self._model.num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            # get the current q values for all actions in state
            current_q = q_s_a[i]
            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + GAMMA * np.amax(q_s_a_d[i])
            x[i] = state
            y[i] = current_q
        self._model.train_batch(self._sess, x, y)

def average_splice(a, n):
    '''
    Return the average values of n splices over collection a
    '''
    result = []
    splice = len(a) / 10
    for i in range(n):
        result.append(np.average(a[int(splice*i):int(splice*(i+1))]))
    return result
        
if __name__ == "__main__":
    random.seed(SEED)
    env = GameState(8, 8, 8, 10)

    num_states = env.cols * env.rows
    num_actions = env.cols * (env.rows - 1) + env.rows * (env.cols - 1)

    model = Model(num_states, num_actions, BATCH_SIZE)
    mem = Memory(50000)

    save_state_file = "AIStates/AIState"
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, save_state_file)
        sess.run(model.var_init)
        gr = GameRunner(sess, model, env, mem, MAX_EPSILON, MIN_EPSILON, LAMBDA)
        num_episodes = 1000
        plot_interval = num_episodes
        cnt = 0
        while cnt < num_episodes:
            gr.run(cnt >= num_episodes - 1)
            cnt += 1
            if cnt % plot_interval == 0:
                saver.save(sess, save_state_file)
                plt.close("all")
                plt.plot(average_splice(gr.reward_store, 10))
                plt.xlim(-1, 11)
                plt.show()

