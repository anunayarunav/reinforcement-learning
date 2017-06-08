import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

class VanillaPG:
    
    def __init__(self, input_shape, layers, num_actions, learning_rate=1e-3):
        
        self.x = tf.placeholder(tf.float32, input_shape)
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.layers = layers
        self.net = self.build_network(layers)
        self.logp = tf.log(self.net)
        self.a = tf.placeholder(tf.int32, shape=(None))
        idx = tf.one_hot(self.a, self.num_actions)
        self.reward = tf.placeholder(tf.float32, shape=(None))
        self.update_variable = tf.reduce_sum(tf.reduce_sum(self.logp*idx, axis=1)*self.reward)
        
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(-self.update_variable)
        
        self.session = tf.Session()
        init_op = tf.global_variables_initializer()

        self.session.run(init_op)
        
    def choose_action(self, s):
        p = self.session.run(self.net, feed_dict= {self.x:[s]})
        return np.random.choice(self.num_actions, size=1, p=p[0])
        
    def update(self, states, actions, rewards):
        self.session.run(self.train_op, feed_dict={self.x:states, self.a:actions, self.reward:rewards})
        
    def build_network(self, layers):
                
        out = self.x
        i = 0
        
        mu=0
        sigma=0.1
        weights = []
            
        nw = "tw"
        nb = "tb"

        while i < len(layers):
            
            if layers[i] == 'conv':
                i += 1
                shape = layers[i]
                conv_w = tf.Variable(tf.truncated_normal(shape=shape, mean=mu, stddev=sigma), name=nw+str(i))
                conv_b = tf.Variable(tf.zeros(shape[-1]), name=nb+str(i))
                out = tf.nn.conv2d(out, conv_w, strides=[1,1,1,1], padding="VALID") + conv_b
                out = tf.nn.relu(out)
                
                weights.append(conv_w)
                weights.append(conv_b)
            
            if layers[i] == 'flatten':
                out = flatten(out)
            
            if layers[i] == 'pool':
                out = tf.nn.max_pool(out, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
         
            if layers[i] == 'fc':
                i += 1
                din = out.get_shape().as_list()[1]
                dout = layers[i][1]

                W = tf.Variable(tf.truncated_normal(shape=[din, dout], mean=mu, stddev=sigma), name=nw+str(i))
                b = tf.Variable(tf.zeros([dout]), name=nb+str(i))
                
                weights.append(W)
                weights.append(b)

                out = tf.matmul(out, W) + b
                out = tf.nn.relu(out)
                
            i += 1
        
        din = out.get_shape().as_list()[1]
        dout = self.num_actions
        W = tf.Variable(tf.truncated_normal(shape=[din, dout], mean=mu, stddev=sigma), name=nw+str(i))
        b = tf.Variable(tf.zeros([dout]), name=nb+str(i))
        
        out = (tf.matmul(out, W) + b)
        probabilities = tf.nn.softmax(out)
        
        return probabilities