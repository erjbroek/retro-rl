import retro
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from train_logging_callback import TrainAndLoggingCallback
from wrappers import ResetEnvWrapper, RewardPrinterWrapper

CHECKPOINT_DIRECTORY = './train/'
LOG_DIRECTORY = './logs/'
callback = TrainAndLoggingCallback(check_freq=200000, save_path=CHECKPOINT_DIRECTORY)
total_steps = 20000
check_count = 4

class MarioAI:
  def __init__(self):
    self.env = retro.make(
      game="SuperMarioBros-Nes",
      scenario="./scenario.json",
      state="./Level1-1.state",
      render_mode=None
    )
    
    self.preprocess()
    self.model = PPO('CnnPolicy', self.env, verbose=1, tensorboard_log=LOG_DIRECTORY, 
                     learning_rate=0.0001, n_steps=512)
    self.loaded = False

  def reset(self, render_mode):
    self.env = retro.make(
      game="SuperMarioBros-Nes",
      scenario="./scenario.json",
      state="./Level1-1.state",
      render_mode=render_mode
    )
    self.preprocess()
  
  def preprocess(self):
    self.env = GrayscaleObservation(self.env, keep_dim=True)
    self.env = DummyVecEnv([lambda: self.env])
    self.env = VecFrameStack(self.env, 4, channels_order='last')

  def train(self):
    if (self.loaded):
      self.model = PPO.load('testmodel')
    self.model.learn(total_timesteps=total_steps / check_count, callback=callback)
    self.model.save('testmodel')
    self.loaded = True

  def run(self):
    self.env.close()
    self.reset(render_mode='human')
    state = self.env.reset()
    done = False
    while not done:
      action, _ = self.model.predict(state)
      obs, reward, done, info = self.env.step(action)

      if done:
        self.env.close()
        self.reset(render_mode=None)

if __name__ == "__main__":
  mario_ai = MarioAI()
  for _ in range(check_count):
    mario_ai.train()
    mario_ai.run()