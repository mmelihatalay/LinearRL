import gym
from gym import spaces
import numpy as np
from collections import deque
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

class LinearModel:
	
	def __init__(self):
		pass
	
class Lynx(LinearModel):
	
	def __init__(self):
		self.state = ["u","w","q","theta","v","p","phi","r"] #m/s, m/s, rad/s, rad, m/s, rad/s, rad, rad/s
		self.input = ["theta0","theta1s","theta1c","theta0T"] #rad, rad, rad, rad
		self.model = {"0":{"A":np.array([[-0.0199,0.0215,0.6674,-9.7837,-0.0205,-0.1600,0.0000,0.0000],
										 [0.0237,-0.3108,0.0134,-0.7215,-0.0028,-0.0054,0.5208,0.0000],
										 [0.0468,0.0055,-1.8954,0.0000,0.0588,0.4562,0.0000,0.0000],
										 [0.0000,0.0000,0.9985,0.0000,0.0000,0.0000,0.0000,0.0532],
										 [0.0207,0.0002,-0.1609,0.0380,-0.0351,-0.6840,9.7697,0.0995],
										 [0.3397,0.0236,-2.6449,0.0000,-0.2715,-10.976,0.0000,-0.0203],
										 [0.0000,0.0000,-0.0039,0.0000,0.0000,1.0000,0.0000,0.0737],
										 [0.0609,0.0089,-0.4766,0.0000,-0.0137,-1.9367,0.0000,-0.2743]]),
						   "B":np.array([[6.9417,-9.2860,2.0164,0.0000],
										 [-93.9179,-0.0020,-0.0003,0.0000],
										 [0.9554,26.4011,-5.7326,0.0000],
										 [0.0000,0.0000,0.0000,0.0000],
										 [-0.3563,-2.0164,-9.2862,3.6770],
										 [7.0476,-33.2120,-152.9537,-0.7358],
										 [0.0000,0.0000,0.0000,0.0000],
										 [17.3054,-5.9909,-27.5911,-9.9111]])},
					  
					  "20":{"A":np.array([[-0.0082,0.0254,-0.0685,-9.7868,-0.0158,-0.1480,0.00000,0.0000],
										  [-0.1723,-0.4346,10.4965,-0.6792,-0.0150,-0.1044,0.45450,0.0000],
										  [0.0417,0.0157,-2.0012,0.0000,0.0482,0.4441,0.00000,0.0000],
										  [0.0000,0.0000,0.9989,0.0000,0.0000,0.0000,0.00000,0.0464],
										  [0.0173,0.0161,-0.1435,0.0311,-0.0604,0.0308,9.77607,-10.1108],
										  [0.1531,0.2739,-2.4044,0.0000,-0.2439,-10.9208,0.00000,-0.0793],
										  [0.0000,0.0000,-0.0032,0.0000,0.0000,1.0000,0.00000,0.0694],
										  [0.0037,0.0455,-0.3753,0.0000,0.0025,-1.9201,0.00000,-0.4404]]),
							"B":np.array([[5.6326,-8.9083,2.0273,0.0000],
										  [-89.9908,-6.0809,0.0010,0.0000],
										  [3.8558,26.6794,-5.7663,0.0000],
										  [0.0000,0.0000,0.0000,0.0000],
										  [0.1249,-2.0098,-9.3275,3.4515],
										  [13.2029,-32.8252,-153.5913,-0.6907],
										  [0.0000,0.0000,0.0000,0.0000],
										  [16.5240,-5.9080,-27.5007,-9.3029]])},
					  
					  "40":{"A":np.array([[-0.0146,0.0347,-0.5681,-9.7934,-0.0083,-0.1321,0.0000,0.0000],
										  [-0.1186,-0.6156,20.6855,-0.5779,-0.0180,-0.2022,0.3519,0.0000],
										  [0.0319,0.0212,-2.1033,0.0000,0.0277,0.4210,0.0000,0.0000],
										  [0.0000,0.0000,0.9994,0.0000,0.0000,0.0000,0.0000,0.0359],
										  [0.0070,0.0184,-0.1303,0.0205,-0.0915,0.5342,9.7869,-20.3077],
										  [-0.0255,0.3040,-2.1361,0.0000,-0.1949,-10.7839,0.0000,-0.1441],
										  [0.0000,0.0000,-0.0021,0.0000,0.0000,1.0000,0.0000,0.0590],
										  [-0.0325,0.0314,-0.2522,0.0000,0.0316,-1.8857,0.0000,-0.68597]]),
							"B":np.array([[4.8686,-8.5123,2.0305,0.0000],
										  [-95.5241,-12.7586,0.0003,0.0000],
										  [7.2883,27.0667,-5.7827,0.0000],
										  [0.0000,0.0000,0.0000,0.0000],
										  [1.1239,-1.8435,-9.3132,3.3289],
										  [27.3295,-30.1532,-153.4552,-0.6662],
										  [0.0000,0.0000,0.0000,0.0000],
										  [15.9423,-5.8252,-27.2699,-8.9726]])},
					  
					  "60":{"A":np.array([[-0.0243,0.0392,-0.6705,-9.8014,-0.0041,-0.1190,0.0000,0.0000],
										  [-0.0467,-0.7285,30.8640,-0.4200,-0.0186,-0.3216,0.3117,0.0000],
										  [0.0280,0.0248,-2.2156,0.0000,0.0159,0.4108,0.0000,0.0000],
										  [0.0000,0.0000,0.9995,0.0000,0.0000,0.0000,0.0000,0.0318],
										  [0.0035,0.0159,-0.1293,0.0133,-0.1228,0.6465,9.7964,-30.5334],
										  [-0.0437,0.2611,-2.0532,0.0000,-0.1713,-10.6565,0.0000,-0.2069],
										  [0.0000,0.0000,-0.0014,0.0000,0.0000,1.0000,0.0000,0.0429],
										  [-0.0273,0.0109,-0.1661,0.0000,0.0529,-1.8568,0.0000,-0.9039]]),
							"B":np.array([[4.6289,-8.0560,2.0386,0.0000],
										  [-107.3896,-21.2288,0.0000,0.0000],
										  [10.7004,27.6889,-5.8115,0.0000],
										  [0.0000,0.0000,0.0000,0.0000],
										  [1.4472,-1.6712,-9.3018,3.7509],
										  [31.4636,-27.4424,-153.3177,-0.7505],
										  [0.0000,0.0000,0.0000,0.0000],
										  [14.5826,-5.9178,-27.0369,-10.1087]])}
					  }
		
class Simulation:
	
	def __init__(self, linearModel, init):
		
		self.keyInt = [int(key) for key in linearModel.model.keys()]
		self.listA = [linearModel.model[key]["A"] for key in linearModel.model.keys()]
		self.listB = [linearModel.model[key]["B"] for key in linearModel.model.keys()]
		self.linearModel = linearModel
		self.init = init
		
	
	def run(self, pilotInput, time, dt=0.01):
		
		if any(np.array(time) < 0):
			raise Exception("Time cannot be less than 0")
			
		#time initilazation    
		t = np.arange(start=time[0], stop=time[-1]+dt, step=dt)
		f = interp1d(time,pilotInput)
		u = f(t)
		dts = np.diff(t)
		
		# A and B initizalitaion for the simulation
		speed = self.init[0]
		A, B = self._findAB(speed)
		
		#since linear model is linearized around equilubrium 
		state0 = np.array(self.init)*0
		
		#init output
		simState = np.zeros([len(t),len(state0)])
		simState[0,:] = self.init
		self.A = []
		self.B = []
		for ind, dt in enumerate(dts):
			x = self._rk4(A, B, state0, dt, u[:,ind:ind+2])
			state0 = x
			A, B = self._findAB(state0[0]+speed)
			simState[ind+1,:] = state0 + self.init
			self.A.append(A)
			self.B.append(B)
			
			
		self.state = pd.DataFrame(columns=self.linearModel.state, data = simState)
		self.time =  t 
		
	def runRL(self, pilotInput0, pilotInput1, state0, speedInit):
		
		speed = state0[0] + speedInit
		A, B = self._findAB(speed)
		u = np.array([list(pilotInput0),list(pilotInput1)]).T
		x = self._rk4(A,B, state0, 0.01, u)
		return x
	
	def _findAB(self, speed):
		speed = speed*0.514444 #m/s to knot since linear model in knots
		if speed < 0:
			return self.listA[0], self.listB[0]
		
		if speed > max(self.keyInt):
			return self.listA[-1], self.listB[-1]
		
		if speed in self.keyInt:
			ind = self.keyInt.index(speed)
			return self.listA[ind], self.listB[ind]
		
		
		findPositive = list(np.array(self.keyInt) > speed)
		upperIndex = findPositive.index(True)
		lowerIndex = upperIndex - 1
		A = self._interp(speed, self.keyInt, self.listA, upperIndex, lowerIndex)
		B = self._interp(speed, self.keyInt, self.listB, upperIndex, lowerIndex)
		return A, B
	
	@classmethod
	def _rk4(cls, A, B, state0, dt, du):
		
		x1 = state0
		u1 = du[:,0]
		xdot1 = cls._linearModelMotion(A, B, x1, u1)
		k1 = dt * xdot1
		
		u2 = 0.5*(du[:,0] + du[:,1])
		x2 = state0+0.5*k1
		xdot2 = cls._linearModelMotion(A, B, x2, u2)
		k2 = dt * xdot2

		u3 = 0.5*(du[:,0] + du[:,1])
		x3 = state0+0.5*k2
		xdot3 = cls._linearModelMotion(A, B, x3, u3)
		k3 = dt * xdot3     
		
		u4 = du[:,1]
		x4 = state0 + k3
		xdot4 = cls._linearModelMotion(A, B, x4, u4)
		k4 = dt * xdot4
		
		xFinal = state0 + k1/6 + k2/3 + k3/3 + k4/6
		return xFinal
	
	@staticmethod
	def _linearModelMotion(A, B, state, control):
		# uses linear model to solve motion
		# xdot = Ax + Bu
		xdotDelta = np.matmul(A, state) + np.matmul(B, control)
		
		return xdotDelta
	
	@staticmethod
	def _interp(targetSpeed, listSpeed, listMatrix, upperIndex, lowerIndex):
		return (listMatrix[upperIndex]*(targetSpeed - listSpeed[lowerIndex]) +
			 listMatrix[lowerIndex]*(-targetSpeed + listSpeed[upperIndex])) / (listSpeed[upperIndex]-listSpeed[lowerIndex])
			
	def plotTime(self, var):
		plt.figure()
		plt.plot(self.time, self.state[var])
		plt.xlabel("Time")
		plt.ylabel(var)
		plt.grid()
		plt.show()

class CustomEnv(gym.Env):
	"""Custom Environment that follows gym interface"""

	def __init__(self):

		super(CustomEnv, self).__init__()
		# Define action and observation space
		# They must be gym.spaces objects
		# Example when using discrete actions:
		self.action_space = spaces.Discrete(8)
		# Example for using image as input (channel-first; channel-last also works):
		self.observation_space = spaces.Box(low=-500, high=500,
											shape=(116,), dtype=np.float64)

	def step(self, action):
		
		self.prev_actions.append(action)

		button_direction = action
		if button_direction == 0:
			pilot_input = np.array([0.01,0,0,0])
		elif button_direction == 1:
			pilot_input = np.array([-0.01,0,0,0])
		elif button_direction == 2:
			pilot_input = np.array([0,0.01,0,0])
		elif button_direction == 3:
			pilot_input = np.array([0,-0.01,0,0])
		elif button_direction == 4:
			pilot_input = np.array([0,0,0.01,0])
		elif button_direction == 5:
			pilot_input = np.array([0,0,-0.01,0])
		elif button_direction == 6:
			pilot_input = np.array([0,0,0,0.01])
		elif button_direction == 7:
			pilot_input = np.array([0,0,0,-0.01])
		self.a += 1
		print(self.a)
		pilot_input += self.prev_input
		print(pilot_input)
		# call helicopter model to make the analysis with given input
		self.state = self.sim.runRL(self.prev_input, pilot_input, self.state, self.state_init[0])
		self.prev_input = pilot_input
		if self.state[0]+self.state_init[0]<20 and self.state[2]<0.01 and self.state[2]>0.01:
			self.done = True
		self.all_state.append(list(self.state))
		self.total_reward = np.linalg.norm(self.state) - np.linalg.norm(self.state_init)
		self.reward = self.total_reward - self.prev_reward
		self.prev_reward = self.total_reward

		if self.done:
			self.reward = -10
		
		info = {}

		u = self.state[0]
		w = self.state[1]
		q = self.state[2]
		theta = self.state[3]
		v = self.state[4]
		p = self.state[5]
		phi = self.state[6]
		r = self.state[7]

		target_delta_u = self.target[0] - u
		target_delta_w = self.target[1] - w
		target_delta_q = self.target[2] - q
		target_delta_theta = self.target[3] - theta
		target_delta_v = self.target[4] - v
		target_delta_p = self.target[5] - p
		target_delta_phi = self.target[6] - phi
		target_delta_r = self.target[7] - r

		observation = [u,w,q,theta,v,p,phi,r,target_delta_u,target_delta_w,target_delta_q,target_delta_theta,target_delta_v,target_delta_p,target_delta_phi,r] + list(self.prev_actions)
		observation = np.array(observation)
		return observation, self.reward, self.done, info

	def reset(self):

		#trim condition, start
		self.state_init = np.array([50,0,0,0,0,0,0,0])
		#creatinb object on the trim condition
		self.sim = Simulation(Lynx(),list(self.state_init))

		# trim is done in delta state
		self.state = self.state_init*0
		#target
		self.target = np.array([20,0,0,0,0,0,0,0])

		self.score = 0
		self.prev_button_direction = 1
		self.button_direction = 1
		self.a = 0
		self.prev_reward = 0

		#previous input to be able to create input for linear model
		self.prev_input = np.array([0,0,0,0])

		self.done = False
		self.all_state = []

		# resolving delta current state
		u = self.state_init[0]
		w = self.state_init[1]
		q = self.state_init[2]
		theta = self.state_init[3]
		v = self.state_init[4]
		p = self.state_init[5]
		phi = self.state_init[6]
		r = self.state_init[7]

		# resolving delta state
		target_delta_u = self.target[0] - u
		target_delta_w = self.target[1] - w
		target_delta_q = self.target[2] - q
		target_delta_theta = self.target[3] - theta
		target_delta_v = self.target[4] - v
		target_delta_p = self.target[5] - p
		target_delta_phi = self.target[6] - phi
		target_delta_r = self.target[7] - r

		time_goal = 1 #sec
		time_len = len(np.arange(0,time_goal,0.01))
		self.prev_actions = deque(maxlen = time_len)
		for i in range(time_len):
			self.prev_actions.append(-1)

		observation = [u,w,q,theta,v,p,phi,r,target_delta_u,target_delta_w,target_delta_q,target_delta_theta,target_delta_v,target_delta_p,target_delta_phi,r] + list(self.prev_actions)
		observation = np.array(observation)
		
		return observation  # reward, done, info can't be included

	#def close (self):