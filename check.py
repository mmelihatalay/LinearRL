from stable_baselines3.common.env_checker import check_env
from trail import CustomEnv
from stable_baselines3 import PPO


env = CustomEnv() 
done = False
obs = env.reset()

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

episodes = 10

for ep in range(episodes):
	obs = env.reset()
	done = False
	while not done:
		action, _states = model.predict(obs)
		obs, rewards, done, info = env.step(action)
		print(rewards)

env.state
#while not done:
#    random_action = env.action_space.sample()
#    obs, reward, done, info = env.step(random_action)
#print(obs)




# It will check your custom environment and output additional warnings if needed
#check_env(env)