import tensorflow as tf
import numpy as np
import os
import gym
import matplotlib.pyplot as plt

#Define hyperparameter
lr = 1e-4     #learning rate
gamma = 0.99  #Discount factor

def preprocess(I):
    I = I[35:195,:]
    I = I[::2,::2,0]
    I[I == 109] = 0
    I[I == 144] = 0
    I[I != 0] = 1
    return np.reshape(I.astype(np.float),(1,6400))

def discounted_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0,r.size)):
        if r[t] != 0:
            running_add = 0
        running_add = running_add*gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

class VanillaPolicy(object):
    def __init__(self,sess,lr):
        self.sess = sess
        self.lr = lr
        self.s = tf.placeholder(shape=[None,6400],dtype=tf.float32,name='state')
        self.dc_r = tf.placeholder(shape=[None,1],dtype=tf.float32,name='discounted_reward')
        #actor
        with tf.variable_scope('actor'):
            w1 = tf.Variable(tf.truncated_normal(shape=[6400,200])/tf.sqrt(6400.0))
            b1 = tf.Variable(tf.zeros(shape=[200],dtype=tf.float32))
            w2 = tf.Variable(tf.truncated_normal(shape=[200,1])/tf.sqrt(200.0))
            b2 = tf.Variable(tf.zeros(shape=[1],dtype=tf.float32))
            h = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.s,w1),b1))
            self.aprob = tf.sigmoid(tf.nn.bias_add(tf.matmul(h,w2),b2))
        #self.batch_aprob = tf.placeholder(shape=[None],dtype=tf.float32)
        self.batch_label = tf.placeholder(shape=[None,1],dtype=tf.float32,name='label')
        self.dlogps = self.batch_label - self.aprob
        with tf.variable_scope('loss'):
            self.loss = self.dlogps*self.dc_r
        with tf.variable_scope('train'):
            self.train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def get_action(self,s):
        return self.sess.run(self.aprob,{self.s:s})
    def update(self,s,label,dc_r):
        self.sess.run(self.train,{self.s:s,self.batch_label:label,self.dc_r:dc_r})
    def saver(self):
        return self.saver

env = gym.make('Pong-v0')
sess = tf.Session()
vp = VanillaPolicy(sess,lr)
writer = tf.summary.FileWriter('vp_logs/',sess.graph)
if os.path.exists('./pong_checkpoint'):
    new_saver = tf.train.import_meta_graph('./pong_checkpoint/my-model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./pong_checkpoint'))
else:
    os.makedirs('./pong_checkpoint')

for ep in range(1000):
    observation = env.reset()
    prev_x = None
    state,label,drs = [],[],[]
    reward_sum = 0
    running_reward = None
    while True:
        env.render()
        current_x = preprocess(observation)
        frame_diff = current_x - prev_x if prev_x is not None else np.zeros((1,6400))
        prev_x = current_x
        state.append(frame_diff)
        aprob = vp.get_action(frame_diff)
        action = 2 if np.random.uniform() < aprob else 3
        y = 1 if action == 2 else 0 # fake label
        label.append(y)
        observation,reward,done,info = env.step(action)
        reward_sum += reward
        drs.append(reward)
        if reward != 0: # pong has either +1 or -1 reward exactly when game ends
            print('ep {}: game finished, reward: {}'.format(ep+1,reward))
        if done:
            #episode_number += 1
            eps = np.vstack(state)
            eplabel = np.vstack(label)
            epr = np.vstack(drs)
            #state,label,drs = [],[],[]
            discounted_epr = discounted_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            vp.update(eps,eplabel,discounted_epr)
            running_reward = reward_sum if running_reward is None else running_reward*0.99 + reward_sum*0.01
            print('resetting env. episode reward total was {} running mean:{}'.format(reward_sum,running_reward))
            vp.saver.save(sess,'./pong_checkpoint/my-model')
            break
