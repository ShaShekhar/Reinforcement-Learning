import tensorflow as tf
import numpy as np
from car_env import CarEnv
import random
import tflearn
from collections import deque

class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.a_out, self.scaled_a_out = self.create_actor_network() # This create actor_network parameters

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_a_out, self.target_scaled_a_out = self.create_actor_network() # this create actor_target_network parameters

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        # tf.gradient(ys,xs,grad_ys)-->It returns a list of Tensor of length len(xs) where each tensor is the sum(dy/dx) for y in ys.
        # grad_ys is a list of tensors of the same length as ys that holds the initial gradients for each y in ys.
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_a_out, self.network_params, self.action_gradient)
        # Calculate the summation for entire batch size
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op   # (- learning rate) for ascent policy
        self.optimize = tf.train.AdamOptimizer(-self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params) # total parameters in actor

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim]) # None is use because the no. of states as input is not fixed
        net = tflearn.fully_connected(inputs, 100)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 80)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        a_out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_a_out = tf.multiply(a_out, self.action_bound)
        return inputs, a_out, scaled_a_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={self.inputs: inputs,self.action_gradient: a_gradient})

    def predict(self, inputs):
        return self.sess.run(self.scaled_a_out, feed_dict={self.inputs: inputs})

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_a_out, feed_dict={self.target_inputs: inputs})

    def update_target_network(self):
        return self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.action_value = self.create_critic_network() # This create critic_network parameters
        # Since all parameters are adding when we create network and we only care about critic_network parameters that's why we slice.
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_action_value = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.action_value)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the action_value w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all actions except for one.
        self.action_grads = tf.gradients(self.action_value, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 100)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 80)
        t2 = tflearn.fully_connected(action, 80)

        net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + (t2.b + t1.b), activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        action_value = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, action_value

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.action_value, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.action_value, feed_dict={self.inputs: inputs,self.action: action})

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_action_value, feed_dict={self.target_inputs: inputs,self.target_action: action})

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={self.inputs: inputs,self.action: actions})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

class OrnsteinUhlenbeckActionNoise(object):
    def __init__(self, mu, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=1):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        #Deque is preferred over list in the cases where we need quicker append and pop operations from both the ends of container,
        #as deque provides an O(1) time complexity for append and pop operations as compared to list which provides O(n) time complexity.
        random.seed(random_seed)

    def add(self, s, a, r, d, s2):
        experience = (s, a, r, d, s2)
        if self.count < self.buffer_size: # Add until the size of buffer
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()          # Delete experience from the top, this happen when buffer_size is full
            self.buffer.append(experience) # Add experience to the end

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []                   # convert deque() into list
        #[([s1], [a1], [r1], [F], [s2]), ([s4], [a4], [r4], [F], [s5]), ([s3], [a3], [r3], [T], [s4]), ([s2], [a2], [r2], [F], [s3]),
        # ([s6], [a6], [r6], [F], [s7]), ([s5], [a5], [r5], [F], [s6])]

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([i[0] for i in batch]) #[[s1]
                                                   #[s4]
                                                   #[s3]
                                                   #[s2]
                                                   #[s6]
                                                   #[s5]]
        a_batch = np.array([i[1] for i in batch])
        r_batch = np.array([i[2] for i in batch])
        d_batch = np.array([i[3] for i in batch])
        s2_batch = np.array([i[4] for i in batch])

        return s_batch, a_batch, r_batch, d_batch, s2_batch

def train(sess, env, actor, critic, action_noise):

    sess.run(tf.global_variables_initializer()) # Initialize all the variable we have created
    writer = tf.summary.FileWriter('logs2', sess.graph)

    # Initialize replay memory
    replay_buffer = ReplayBuffer(1000000,1)

    for i in range(300):

        s = env.reset()

        ep_reward = 0 # episode total reward
        time_steps = 0 # count the number of time steps in one episode until terminate

        for j in range(600):

            if True:
                env.render()
            # Added exploration noise
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + action_noise()

            s2, r, done = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              done, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > 64:
                s_batch, a_batch, r_batch, d_batch, s2_batch = replay_buffer.sample_batch(64)

                # Calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(64):
                    if d_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                critic.train(s_batch, a_batch, np.reshape(y_i, (64,1)))

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r
            time_steps += 1

            if done:
                print('| Reward: {} | Episode: {} | Time_steps: {}'.format(int(ep_reward), i, (time_steps)))
                break

def main():

    with tf.Session() as sess:

        env = CarEnv()
        np.random.seed(1)
        tf.set_random_seed(1)

        state_dim = env.state_dim
        action_dim = env.action_dim
        action_bound = env.action_bound_high

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,0.001,0.01,64)

        critic = CriticNetwork(sess, state_dim, action_dim,0.001,0.01,0.9,actor.get_num_trainable_vars())

        action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        train(sess, env, actor, critic, action_noise)


if __name__ == '__main__':
    main()
