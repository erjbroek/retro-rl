import retro
import time
import os
import random
from utils import format_time_from_seconds
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from train_logging_callback import TrainAndLoggingCallback
from wrappers import FrameSkip
from pathlib import Path

BK2_DIRECTORY = os.path.abspath("./recordings/bk2")
LOG_DIRECTORY = os.path.abspath('./logs')
STATE_DIRECTORY = os.path.abspath('./states')

total_steps = 100_000
check_count = 40
callback = TrainAndLoggingCallback(check_freq=total_steps / 6, save_path=BK2_DIRECTORY)
start_time = time.time()

class MarioAI:
  def __init__(self):
    self.state_files = [str(os.path.join(STATE_DIRECTORY, file)) for file in os.listdir(STATE_DIRECTORY)][:4] # for now there are 4 states, but if i forget this in the future, ill limit it to 4
    self.env = retro.make(
      game="SuperMarioBros-Nes",
      scenario="./scenario.json",
      state="./Level1-1.state",
      render_mode=None,
    )
    
    self.preprocess()
    self.model = PPO('CnnPolicy', self.env, verbose=1, tensorboard_log=LOG_DIRECTORY, 
                     learning_rate=0.0002, n_steps=512)
    self.saved = False
    self.current_round = 0
    self.times_saved = 0
    self.visualisation_freq = 2
    self.saving_frequency = 10

  # because the environments render mode cannot be changed outside of initialisation,
  # this reset function resets the environment so that the render_mode can be changed
  def reset(self, render_mode, should_record):
    make_kwargs = {
      "game": "SuperMarioBros-Nes",
      "scenario": "./scenario.json",
      "state": "./Level1-1.state",
      "render_mode": render_mode
    }

    if not should_record:
      make_kwargs["record"] = BK2_DIRECTORY
    # else:
      # make_kwargs["state"] = random.choice(self.state_files)
      print(f"Current state: {make_kwargs["state"].split('/')[-1]}")
      print(f"Round {self.current_round}")
      print(f"Estimated time left: {format_time_from_seconds(round(time.time() - start_time, 2))} / {format_time_from_seconds(round(((time.time() - start_time) / self.current_round) * check_count))}")

    self.env = retro.make(**make_kwargs)
    self.env.movie_id = self.current_round
    self.preprocess()
  
  # preprocesses the environment.
  # - GrayscaleObservation turns the rgb environment gray scaled:
  #     This reduces the input dimensionality (from 3 color channels to 1), which speeds up training and 
  #     reduces memory usage. Most tasks don’t need color information, so this is often sufficient. An
  #     visualisation of this is found in the assets directory state.png.
  #
  # - DummyVecEnv wraps the environment for vectorized operations:
  #     Many RL libraries expect environments to support batch processing (multiple environments at once),
  #     even if you only use one. DummyVecEnv provides this interface for a single environment,
  #     which makes sure it's compatible
  # .
  # - VecFrameStack stacks 4 consecutive frames for temporal information:
  #     Stacks several consecutive frames (In this case 4) together, helping it 'see' motion (since a 
  #     single frame doesn’t show movement). Example, in games, knowing the last few frames helps the
  #     agent understand object velocities and directions. Example of this is found in the assets 
  #     directory observation3.png, showing the 4 consecutive frames.
  def preprocess(self):
    self.env = GrayscaleObservation(self.env, keep_dim=True)
    self.env = FrameSkip(self.env, skip=4)
    self.env = DummyVecEnv([lambda: self.env])
    self.env = VecFrameStack(self.env, 4, channels_order='last')

  # This is what actually loads and trains the model
  #   Here, the training is split up into check_count number of ste.ps, which enables us to visualise
  #   the agent acting in the environment every once in a while
  def train(self):
    if (self.saved):
      self.model = PPO.load('SMB_PPO')
      self.model.set_env(self.env)
      print('model loaded succesfully.')

    self.model.learn(total_timesteps=total_steps / check_count, callback=callback)
    print('done training')

    # store the models during different stages of training, which enables us to see progress made
    if self.current_round + 1 % self.saving_frequency == 0:
      self.model.save(f"SMB_PPO{total_steps / check_count * self.current_round}")
      self.times_saved += 1

    self.model.save('SMB_PPO')
    print('model saved succesfully')
    print()
    self.saved = True
    self.current_round += 1

  # Visualises the agent in the environment. Important to mention this is seperate for training the
  # agent, and is just run to see mario playing the game.
  def run(self):
    self.env.close()
    if (self.current_round % self.visualisation_freq == 0):
      self.reset(render_mode='human', should_record=True)
      state = self.env.reset()
      finished = False

      while not finished:
        action, _ = self.model.predict(state)
        obs, reward, done, info = self.env.step(action)

        if info[0]['time'] <= 250 or done[0]:
          finished = True
          self.env.close()
    
    self.reset(render_mode=None, should_record=False)

if __name__ == "__main__":
  mario_ai = MarioAI()
  for _ in range(check_count):
    mario_ai.train()
    mario_ai.run()