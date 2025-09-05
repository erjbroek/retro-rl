import retro


def main():
  env = retro.make(
    game="SuperMarioBros-Nes",
    scenario="./scenario.json",
    state="./Level3-1.state"
  )
  env.reset()
  
  # print(env.buttons)
  while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(reward)
    env.render()
    if terminated or truncated:
      env.reset()

  env.close()


if __name__ == "__main__":
  main()