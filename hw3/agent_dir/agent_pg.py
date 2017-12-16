from agent_dir.agent import Agent
import scipy
import numpy as np
from multiprocessing import Queue

import time
from collections import deque
from agent_dir.Config import Config
from agent_dir.Environment import Environment
from agent_dir.NetworkVP import NetworkVP
from agent_dir.ProcessAgent import ProcessAgent
from agent_dir.ProcessStats import ProcessStats
from agent_dir.ThreadDynamicAdjustment import ThreadDynamicAdjustment
from agent_dir.ThreadPredictor import ThreadPredictor
from agent_dir.ThreadTrainer import ThreadTrainer

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        self.stats = ProcessStats()
        self.training_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        self.prediction_q = Queue(maxsize=Config.MAX_QUEUE_SIZE)
        
        self.env2 = env
        self.model = NetworkVP(Config.DEVICE, Config.NETWORK_NAME, self.env2.action_space.n)
        if Config.LOAD_CHECKPOINT:
            self.stats.episode_count.value = self.model.load()

        self.training_step = 0
        self.frame_counter = 0

        self.agents = []
        self.predictors = []
        self.trainers = []
        self.dynamic_adjustment = ThreadDynamicAdjustment(self)

        if args.test_pg:
            #you can load your model here
            self.model.load()
            print('loading trained model')

    def add_agent(self):
        self.agents.append(
            ProcessAgent(len(self.agents), self.prediction_q, self.training_q, self.stats.episode_log_q))
        self.agents[-1].start()

    def remove_agent(self):
        self.agents[-1].exit_flag.value = True
        self.agents[-1].join()
        self.agents.pop()

    def add_predictor(self):
        self.predictors.append(ThreadPredictor(self, len(self.predictors)))
        self.predictors[-1].start()

    def remove_predictor(self):
        self.predictors[-1].exit_flag = True
        self.predictors[-1].join()
        self.predictors.pop()

    def add_trainer(self):
        self.trainers.append(ThreadTrainer(self, len(self.trainers)))
        self.trainers[-1].start()

    def remove_trainer(self):
        self.trainers[-1].exit_flag = True
        self.trainers[-1].join()
        self.trainers.pop()

    def train_model(self, x_, r_, a_, trainer_id):
        self.model.train(x_, r_, a_, trainer_id)
        self.training_step += 1
        self.frame_counter += x_.shape[0]

        self.stats.training_count.value += 1
        self.dynamic_adjustment.temporal_training_count += 1

        if Config.TENSORBOARD and self.stats.training_count.value % Config.TENSORBOARD_UPDATE_FREQUENCY == 0:
            self.model.log(x_, r_, a_)

    def save_model(self):
        self.model.save(self.stats.episode_count.value)


    def init_game_setting(self):
        """
        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary
        """
        ##################
        # YOUR CODE HERE #
        ##################
        
        self.num_actions = self.env2.action_space.n
        self.actions = np.arange(self.num_actions)
        self.nb_frames = Config.STACKED_FRAMES
        self.frame_q = deque(maxlen=self.nb_frames)

    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.env = Environment()
		
        self.stats.start()
        self.dynamic_adjustment.start()

        learning_rate_multiplier = (
                                       Config.LEARNING_RATE_END - Config.LEARNING_RATE_START) / Config.ANNEALING_EPISODE_COUNT
        beta_multiplier = (Config.BETA_END - Config.BETA_START) / Config.ANNEALING_EPISODE_COUNT

        while self.stats.episode_count.value < Config.EPISODES:
            step = min(self.stats.episode_count.value, Config.ANNEALING_EPISODE_COUNT - 1)
            self.model.learning_rate = Config.LEARNING_RATE_START + learning_rate_multiplier * step
            self.model.beta = Config.BETA_START + beta_multiplier * step

            # Saving is async - even if we start saving at a given episode, we may save the model at a later episode
            if Config.SAVE_MODELS and self.stats.should_save_model.value > 0:
                self.save_model()
                self.stats.should_save_model.value = 0

            time.sleep(0.01)

        self.dynamic_adjustment.exit_flag = True
        while self.agents:
            self.remove_agent()
        while self.predictors:
            self.remove_predictor()
        while self.trainers:
            self.remove_trainer()


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent
        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        frame=Environment._preprocess(observation)
        self.frame_q.append(frame)
        if(len(self.frame_q)==4):
            x_ = np.array(self.frame_q)
            x_ = np.transpose(x_, [1, 2, 0])
            prediction=self.model.predict_single(x_)
            action = np.random.choice(self.actions, p=prediction)
        else:
            action = 0
        return action

