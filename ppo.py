import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

EP_MAX = 500
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM,A_DIM = 3,1

METHOD = {'name':'clip','epsilon':0.2}

class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.s = tf.placeholder(tf.float32,[None,S_DIM],'state')
        #critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.s,100,tf.nn.relu)
            self.v = tf.layers.dense(l1,1)
            self.dc_r = tf.placeholder(tf.float32,[None,1],'discounted_reward')
            self.advantage = self.dc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)
        #actor
        pi,pi_params = self._build_anet('pi',trainable=True)
        oldpi,oldpi_params = self._build_anet('oldpi',trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1),axis=0)
        with tf.variable_scope('update_oldpi'):
            self.update_old_pi = [oldp.assign(p) for p,oldp in zip(pi_params,oldpi_params)]
        self.a = tf.placeholder(tf.float32,[None,A_DIM],'action')
        self.adv = tf.placeholder(tf.float32,[None,1],'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = pi.prob(self.a)/oldpi.prob(self.a)
                surrogate = ratio*self.adv
            self.aloss = -tf.reduce_mean(tf.minimum(surrogate,tf.clip_by_value(ratio,1-METHOD['epsilon'],1+METHOD['epsilon'])*self.adv))
        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        tf.summary.FileWriter('logs/',self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def update(self,s,a,r):
        self.sess.run(self.update_old_pi)
        adv = self.sess.run(self.advantage,{self.s:s,self.dc_r:r})
        #update actor
        [self.sess.run(self.atrain_op,{self.s:s,self.a:a,self.adv:adv}) for _ in range(A_UPDATE_STEPS)]
        #update critic
        [self.sess.run(self.ctrain_op,{self.s:s,self.dc_r:r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self,name,trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.s,100,tf.nn.relu,trainable=trainable)
            mu = 2*tf.layers.dense(l1,A_DIM,tf.nn.tanh,trainable=trainable)
            sigma = tf.layers.dense(l1,A_DIM,tf.nn.softplus,trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu,scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=name)
        return norm_dist,params

    def choose_action(self,s):
        s = s[np.newaxis,:]
        a = self.sess.run(self.sample_op,{self.s:s})[0]
        return np.clip(a,-2,2)
    def get_v(self,s):
        if s.ndim < 2: s = s[np.newaxis,:]
        return self.sess.run(self.v,{self.s:s})[0,0]

env = gym.make('Pendulum-v0')
ppo = PPO()
all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s,buffer_a,buffer_r = [],[],[]
    ep_r = 0
    t = 0
    while True:
        t += 1
        env.render()
        a = ppo.choose_action(s)
        s_,r,done,_ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)
        s = s_
        ep_r += r
        #update ppo
        if (t)%BATCH == 0 or t == 200:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA*v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()
            bs,ba,br = np.vstack(buffer_s),np.vstack(buffer_a),np.array(discounted_r)[:,np.newaxis]
            buffer_s,buffer_a,buffer_r = [],[],[]
            ppo.update(bs,ba,br)
        if done:
            break
    if ep == 0:
        all_ep_r.append(ep_r)
    else:
        all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print('Ep:{} Ep_r:{}'.format(ep,ep_r))

plt.plot(np.arange(len(all_ep_r)),all_ep_r)
plt.xlabel('Episode'),plt.ylabel('Moving Average Episode Reward')
plt.show()
