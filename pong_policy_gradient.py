import numpy as np
import gym
import pickle
import os
import matplotlib.pyplot as plt

# hyperparameters
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = False # resume from previous checkpoint?

# model initialization
D = 80 * 80 # input dimensionality: 80x80 grid
if os.path.exists('weights.pkl'):
  weights = pickle.load(open('weights.pkl', 'rb'))
else:
  weights = {}
  weights['W1'] = np.random.randn(D,H) / np.sqrt(D) # "Xavier" initialization
  weights['W2'] = np.random.randn(H,1) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in weights.items()} # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in weights.items()} # rmsprop memory

def preprocess(I):
    #np.set_printoptions(threshold=np.nan)
    I = I[35:195,:]
    I = I[::2,::2,0]
    I[I == 109] = 0
    I[I == 144] = 0
    I[I != 0] = 1
    return np.reshape(I.astype(np.float),(1,6400))

def policy_forward(x):
    h = np.dot(x,weights['W1'])#+biases['b1']
    h[h<0] = 0
    logp = np.dot(h,weights['W2'])#+biases['b2']
    p = 1.0/(1.0+np.exp(-logp))
    return p,h

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0,r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add*gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def policy_backward(eph,epdlogp):
    # eph is an array of intermediate hidden state
    dW2 = np.dot(eph.T,epdlogp)
    dh = np.dot(epdlogp,weights['W2'].T)
    dh[eph <= 0] = 0 # backprop relu
    dW1 = np.dot(epx.T,dh)
    return {'W1':dW1,'W2':dW2}

env = gym.make('Pong-v0')
observation = env.reset()
prev_x = None # used in computing the difference frame
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 0
while True:
    env.render()
    current_x = preprocess(observation)
    x = current_x - prev_x if prev_x is not None else np.zeros((1,6400))
    prev_x = current_x
    aprob,h = policy_forward(x)
    action = 2 if np.random.uniform() < aprob else 3 # roll the dice unbiased
    # record various intermediates(needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hiddenstate
    y = 1 if action == 2 else 0 # a 'fake label'
    dlogps.append(y - aprob) # grad that encourages the action that was taken to be taken
    observation,reward,done,info = env.step(action)
    reward_sum += reward
    drs.append(reward)
    if done: # an episode finished
        episode_number += 1
        # stack together all inputs,hidden states,action gradients, and reward for this episode
        epx = np.vstack(xs)
        #print(epx.shape) #(1147,6400)
        eph = np.vstack(hs)
        #print(eph.shape)#(1147,200)
        epdlogp = np.vstack(dlogps)
        #print(epdlogp.shape)#(1147,1)
        epr = np.vstack(drs)
        #print(epr.shape)#(1147,1)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        # compute the discounted reward backward through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal(helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage(PG magic happens right here)
        grad = policy_backward(eph,epdlogp)
        for k in weights:
            grad_buffer[k] += grad[k] # accumulate grad over batch
        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k,v in weights.items():
                g = grad_buffer[k]/batch_size # gradient
                #np.set_printoptions(threshold=np.nan)
                #print('gradient value', g)
                rmsprop_cache[k] = decay_rate*rmsprop_cache[k] + (1 - decay_rate)*g**2
                weights[k] += learning_rate*g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
        # print statements
        running_reward = reward_sum if running_reward is None else running_reward*0.99 + reward_sum*0.01
        print('resetting env. episode reward total was {} running mean:{}'.format(reward_sum,running_reward))
        if episode_number % 10 == 0:
            pickle.dump(weights,open('weights.pkl','wb'))
        reward_sum = 0
        observation = env.reset()
        prex_x = None

    if reward != 0: # pong has either +1 or -1 reward exactly when game ends
        print('ep {}: game finished, reward: {}'.format(episode_number+1,reward))# + (''if reward == -1 else '!!!!!!!')
