import gymnasium as gym

class RewardPrinterWrapper(gym.Wrapper):
  def step(self, action):
    observation, reward, terminated, truncated, info = self.env.step(action)
    print(f"reward={reward:,.2f} (terminated={terminated}, truncated={truncated})")
    return observation, reward, terminated, truncated, info
  
class FrameSkip(gym.Wrapper):
  def __init__(self, env, skip=4):
    super().__init__(env)
    self.env = env
    self.skip = skip
    if skip < 1:
      raise ValueError(f"Invalid skip value: {skip}")

  def step(self, action):
    reward = 0
    for _ in range(self.skip):
      obs, rew, terminated, truncated, info = self.env.step(action)
      reward += rew

    return obs, reward, terminated, truncated, info
