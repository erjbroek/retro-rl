import retro
from gymnasium.wrappers import GrayscaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3 import PPO
from train_logging_callback import TrainAndLoggingCallback

CHECKPOINT_DIRECTORY = './train/'
LOG_DIRECTORY = './logs/'
callback = TrainAndLoggingCallback(check_freq=200000, save_path=CHECKPOINT_DIRECTORY)
total_steps = 2000
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

  def reset(self, render_mode, should_record):
    record_dir = '.' if should_record else None
    self.env = retro.make(
      game="SuperMarioBros-Nes",
      scenario="./scenario.json",
      state="./Level1-1.state",
      render_mode=render_mode,
      record=record_dir
    )
    self.preprocess()
  
  def preprocess(self):
    self.env = GrayscaleObservation(self.env, keep_dim=True)
    self.env = DummyVecEnv([lambda: self.env])
    self.env = VecFrameStack(self.env, 4, channels_order='last')

  def train(self):
    if (self.loaded):
      self.model = PPO.load('testmodel')
      self.model.set_env(self.env)
    self.model.learn(total_timesteps=total_steps / check_count, callback=callback)
    self.model.save('testmodel')
    self.loaded = True

  def run(self):
    self.env.close()
    self.reset(render_mode='human', should_record=True)
    state = self.env.reset()
    finished = False
    while not finished:
      action, _ = self.model.predict(state)
      obs, reward, done, info = self.env.step(action)
      if info[0]['time'] <= 300:
        finished = True
        self.env.close()
        self.reset(render_mode=None, should_record=False)

if __name__ == "__main__":
  mario_ai = MarioAI()
  for _ in range(check_count):
    mario_ai.train()
    mario_ai.run()