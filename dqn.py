import gym
import itertools
import numpy as np
import random
import tensorflow as tf
from collections import namedtuple

#sess = tf.Session()

env = gym.make('Breakout-v0')
#print(env.action_space.n)
VALID_ACTIONS = [0,1,2,3]

class StateProcessor():
    def __init__(self):
        with tf.variable_scope('state_preprocess'):
            self.input_state = tf.placeholder(shape=[210,160,3],dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output,34,0,160,160)
            self.output = tf.image.resize_images(self.output,[84,84],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)
    def process(self,sess,state):
        return sess.run(self.output, feed_dict={self.input_state:state})


class Estimator():
    def __init__(self,scope='estimator'):
        self.scope = scope
        with tf.variable_scope(scope):
            self._build_model()

    def _build_model(self):
        # build the tensorflow graph
        self.x_pl = tf.placeholder(shape=[None,84,84,4],dtype=tf.uint8,name='X')
        self.y_pl = tf.placeholder(shape=[None],dtype=tf.float32,name='y')
        self.action_pl = tf.placeholder(shape=[None],dtype=tf.int32,name='actions')
        X = tf.to_float(self.x_pl) / 255.0
        batch_size = tf.shape(self.x_pl)[0]
        # Three convolutional layers
        conv1 = tf.contrib.layers.conv2d(X,32,8,4,activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(conv1,64,4,2,activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(conv2,64,3,1,activation_fn=tf.nn.relu)
        # fully connected layers
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened,512)
        self.predictions = tf.contrib.layers.fully_connected(fc1,len(VALID_ACTIONS))

        with tf.variable_scope('q_eval'):
            # get the predictions for chosen actions only
            gather_indices = tf.range(batch_size)*tf.shape(self.predictions)[1] + self.actions_pl
            self.action_predictions = tf.gather(tf.reshape(self.predictions,[-1]),gather_indices)
        with tf.variable_scope('loss'):
            # calculate the loss
            self.loss = tf.reduce_mean(tf.squared_difference(self.y_pl,self.action_predictions))
        with tf.variable_scope('train'):
            self.optimizer = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6)
            self.train_op = self.optimizer.minimize(self.loss,global_step=tf.contrib.framework.get_global_step())

    def predict(self,sess,s):
        return sess.run(self.predictions,feed_dict={self.x_pl:s})

    def update(self,sess,s,a,y):
        feed_dict = {self.x_pl:s,self.y_pl:y,self.action_pl:a}
        global_step, _ ,loss = sess.run([tf.contrib.framework.get_global_step(),self.train_op,self.loss],feed_dict)
        return loss

class ModelParametersCopier():
    def __init__(self,estimator1,estimator2):
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='estimator1') #target_net
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='estimator2') #eval_net
        with tf.variable_scope('param_replacement'):
            self.target_replace_op = [tf.assign(t,e) for t,e in zip(t_params,e_params)]

    def make(self,sess):
        sess.run(self.target_replace_op)

def make_epsilon_greedy_policy(estimator, nA):
    def policy_fn(sess, observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def deep_q_learning(sess,env,q_estimator,target_estimator,
                     state_processor,num_episodes,replay_memory_size=500000,
                     update_target_estimator_every=10000,
                     discount_factor=0.99,
                     epsilon_start=1.0,
                     epsilon_end=0.1,
                     epsilon_decay_steps=500000,
                     batch_size=32):
    Transition = namedtuple('Transition',['state','action','reward','done'])
    # The replay memory
    replay_memory = []
    # Make model copier object
    estimator_copy = ModelParametersCopier(target_estimator,q_estimator)
    # Get the current time step
    total_t = sess.run(tf.contrib.framework.get_global_step())
    # The epsilon decay schedule
    epsilons = np.linspace(epsilon_start,epsilon_end,epsilon_decay_steps)
    # The policy we're following
    policy = make_epsilon_greedy_policy(q_estimator,len(VALID_ACTIONS))

    for i_episode in range(num_episodes):
        # Reset the environment
        state = env.reset()
        state = state_processor.process(sess,state)
        state = np.stack([state]*4,axis=2)
        loss = None

        # One step in the environment
        for t in itertools.count():
            env.render()
            # Epsilon for this time step
            epsilon = epsilons[min(total_t,epsilon_decay_steps-1)]
            # Maybe update the target estimator
            if total_t % update_target_estimator_every == 0:
                estomator_copy.make(sess)
                print('\nCopied model parameters to target network.')
            # Print out which step we're on,useful for debugging.
            print('\rStep {} ({}) @ Episode {}/{}, loss: {}'.format(t,total_t,i_episode+1,num_epsodes,loss),end='')

            # Take a step
            action_probs = policy(sess,state,epsilon)
            action = np.random.choice(np.arange(len(action_probs)),p=action_probs)
            next_state,reward,done,_ = env.step(VALID_ACTIONS[action])
            next_state = state_processor.process(sess,next_state)
            next_state = np.append(state[:,:,1:],np.expand_dims(next_state,2), axis=2)

            # If our replay memory is full,pop the first element
            if len(replay_memory) == replay_memory_size:
                replay_memory.pop(0)
            # Save transition to replay memory
            replay_memory.append(Transition(state,action,reward,next_state,done))
            if len(replay_memory) > batch_size:
                # Sample a minibatch from the replay memory
                samples = random.sample(replay_memory,batch_size)
                states_batch, action_batch, reward_batch, next_state_batch, done_batch = map(np.array,zip(*samples))

                # Calculate q values and targets
                q_value_next = target_estimator.predict(sess,next_states_batch)
                targets_batch = reward_batch + np.invert(done_batch).astype(np.float32)*discount_factor*np.amax(q_values_next,axis=1)

                # Perform gradient descent update
                states_batch = np.array(states_batch)
                loss = q_estimator.update(sess,states_batch,action_batch,targets_batch)
            if done:
                break
            state = next_state
            total_t += 1

    tf.reset_default_graph()        

# Create a global step variable
global_step = tf.Variable(0,name='global_step',trainable=False)

# Create estimators
q_estimator = Estimator(scope='q_estimator')
target_estimator = Estimator(scope='target_q')

# State pocessor
state_processor = StateProcessor()

# Run it!
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    deep_q_learning(sess,env,
                    q_estimator=q_estimator,
                    target_estimator=target_estimator,
                    state_processor=state_processor,
                    num_episodes=10000,
                    replay_memory_size=50,
                    update_target_estimator_every=10000,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=500000,
                    discount_factor=0.99,
                    batch_size=32)
