import retro
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from callbacks.train_logging_callback import TrainAndLoggingCallback

CHECKPOINT_DIRECTORY = './train/'
LOG_DIRECTORY = './logs/'
callback = TrainAndLoggingCallback(check_freq=20000, save_path=CHECKPOINT_DIRECTORY)

class MarioAI:
  def __init__(self):
    self.env = retro.make(
      game="SuperMarioBros-Nes",
      scenario="./scenario.json",
      state="./Level1-1.state"
    )
    
    self.preprocess()

  def preprocess(self):
    self.env = GrayscaleObservation(self.env, keep_dim=True)
    self.env = DummyVecEnv([lambda: self.env])
    self.env = VecFrameStack(self.env, 4, channels_order='last')

  def run(self):
    self.env.reset()
    while True:
      action = self.env.action_space.sample()
      obs, reward, done, info = self.env.step([action])

      self.env.render()
      if done:
        self.env.reset()


if __name__ == "__main__":
  mario_ai = MarioAI()
  mario_ai.run()