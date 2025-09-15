import gymnasium as gym

class RewardPrinterWrapper(gym.Wrapper):
  def step(self, action):
    observation, reward, terminated, truncated, info = self.env.step(action)
    print(f"reward={reward:,.2f} (terminated={terminated}, truncated={truncated})")
    return observation, reward, terminated, truncated, info
