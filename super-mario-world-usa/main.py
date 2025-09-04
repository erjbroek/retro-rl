import retro

def main():  
  env = retro.make(
    game='SuperMarioWorld-Snes',
    scenario="./scenario.json",
  )
  env.reset()
  while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    if info["lives"] == 3:
      env.reset()

if __name__ == "__main__":
  main()
