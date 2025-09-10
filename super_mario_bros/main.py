import retro


class MarioAI:
  def __init__(self):
    self.env = retro.make(
      game="SuperMarioBros-Nes",
      scenario="./scenario.json",
      state="./Level1-1.state"
    )

  def preprocess(self, observation):
    pass

  def run(self):
    self.env.reset()
    while True:
      action = self.env.action_space.sample()
      observation, reward, terminated, truncated, info = self.env.step(action)

      self.env.render()
      if terminated or truncated:
        self.env.reset()


if __name__ == "__main__":
  mario_ai = MarioAI()
  mario_ai.run()