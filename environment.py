import gym
env = gym.make('Taxi-v3').env
env.render()

env.reset()
env.render()

print("Action Space: ", env.action_space)
print("State Space: ", env.observation_space)

# making a custom state
state = env.encode(3,1,2,0)
print(state)
env.s = state
env.render()

# reward Table
print(env.P[328])