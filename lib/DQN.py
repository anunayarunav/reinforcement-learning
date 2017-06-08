import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

class DQN:
    
    def __init__(self, config):
        
        self.input_shape = config['input_shape']
        self.action_count = config['action_count']
        
        self.x = tf.placeholder(tf.float32, shape=self.input_shape)

        self.discount = config['discount']
        self.layers = config['layers']
        self.learning_rate = config['lr']
        self.reg = config['reg']
        
        self.network = self.build_network(self.layers)
        self.target = self.build_network(self.layers, True)
        
        #variables
        self.r = tf.placeholder(tf.float32, shape=None)
        self.f = tf.placeholder(tf.float32, shape=None)
        self.y = self.r + self.f*self.discount*tf.reduce_max(self.target, axis=1)
        
        self.a = tf.placeholder(tf.int32, shape=(None))
        idx = tf.one_hot(self.a, self.action_count)
        
        self.z = tf.reduce_sum(self.network*idx, axis=1)
        self.y2 = tf.placeholder(tf.float32, shape=(None))
        self.e = tf.reduce_sum(self.z-self.y2)**2
        self.add_reg(self.reg)
        
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.e)
        
        self.action = tf.argmax(self.network, axis=1)

        self.update = config['update']
        
        self.session = tf.Session()
        init_op = tf.global_variables_initializer()
        
        self.session.run(init_op)
        
        self.count = 0
        
    def max_q(self, s):
        return self.session.run(self.action, feed_dict={self.x:np.array([s])})
        
    def play(self, X):
        
        if self.count%self.update == 0:
            print("Updating target")
            self.update_target()
        
        self.count += 1
        
        s, a, r, sn, f = X
        
        y = self.session.run(self.y, feed_dict={self.x:sn, self.r:r, self.f:f})
        
        self.session.run(self.train_op, feed_dict={self.x:s, self.y2:y, self.a:a})
        
        return self.session.run(self.e, feed_dict={self.x:s, self.y2:y, self.a:a})
        
    def update_target(self):
        for w1, w2 in zip(self.network_weights, self.target_weights):
            op = tf.assign(w2, w1)
            self.session.run(op)
    
    def add_reg(self, reg):
        for w in self.network_weights:
            self.e += reg*tf.reduce_sum(w**2)
    
    def build_network(self, layers, target=False):
        
        out = self.x
        i = 0
        
        mu=0
        sigma=0.1
        weights = []
            
        if target:
            nw = "tw"
            nb = "tb"
            
        else:
            nw = "w"
            nb = "b"
        
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
        dout = self.action_count
        W = tf.Variable(tf.truncated_normal(shape=[din, dout], mean=mu, stddev=sigma), name=nw+str(i))
        b = tf.Variable(tf.zeros([dout]), name=nb+str(i))

        weights.append(W)
        weights.append(b)
        
        out = (tf.matmul(out, W) + b)
        
        if target:
            self.target_weights = weights
        else:
            self.network_weights = weights
        
        return out