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

  # because the environments render mode cannot be changed outside of initialisation,
  # this reset function resets the environment so that the render_mode can be changed
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
    self.env = DummyVecEnv([lambda: self.env])
    self.env = VecFrameStack(self.env, 4, channels_order='last')

  # This is what actually loads and trains the model
  #   Here, the training is split up into check_count number of ste.ps, which enables us to visualise
  #   the agent acting in the environment every once in a while
  def train(self):
    if (self.loaded):
      self.model = PPO.load('SMB_PPO')
      self.model.set_env(self.env)
    self.model.learn(total_timesteps=total_steps / check_count, callback=callback)
    self.model.save('SMB_PPO')
    self.loaded = True

  # Visualises the agent in the environment. Important to mention this is seperate for training the
  # agent, and is just run to see mario playing the game.
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