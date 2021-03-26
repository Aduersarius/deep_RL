# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division

import os
import optparse
from utils import mkdir_p

from sampler import *
from agents import *
from emulator import *
from simulators import *
from visualizer import *
from data import *


def main():
    """
    it is recommended to generate database using sampler.py before run main
    """

    """
    Parse command line options and start the server.
    """

    model_type = 'LSTM'

    n_episode_training = 1000
    n_episode_testing = 100

    # db_type = 'SinSamplerDB'; db = 'concat_half_base_'; Sampler = SinSampler
    #db_type = 'PairSamplerDB'
    #db = 'randjump_100,1(10, 30)[]_'
    #fld = os.path.join('..', 'data', db_type, db + 'A')
    #print('data folder: {}'.format(fld))
    #sampler = PairSampler('load', fld=fld)
    sampler = get_data(ticker='SPCE', feed_window=1000, prediction_window=0, size=1, only_price=True)
    trading_cost = 3
    window_state = 40
    env = Market(sampler, window_state, trading_cost)

    # model param
    batch_size = 32
    learning_rate = 1e-4
    discount_factor = 0.96
    exploration_min = 0.01
    agent = DQNAgent(learning_rate=learning_rate, batch_size=batch_size,
                     discount_factor=discount_factor, epsilon_min=exploration_min, epsilon_decay=0.995)
    agent.build_model(model_type, env)

    visualizer = Visualizer(env.action_labels)

    # fld_save = os.path.join(opts.run, sampler.title, model.model_name,
    #                         str((env.window_state, sampler.window_episode, agent.batch_size, learning_rate,
    #                              agent.discount_factor, exploration_decay, env.trading_cost)))


    simulator = Simulator(agent, env, visualizer=visualizer)

    #if opts.mode == 'train':
    simulator.train(n_episode_training, save_per_episode=20)
    #else:
       # agent.model = restore_model(os.path.join(fld_save, 'model'), learning_rate)

    # print('='*20+'\nin-sample testing\n'+'='*20)
    simulator.test(n_episode_testing, save_per_episode=20, subfld='in-sample testing')
    simulator.test(n_episode_testing, save_per_episode=20, subfld='out-of-sample testing')


if __name__ == '__main__':
    main()
