import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

class TRPO:
    
    def __init__(self, input_shape, layers, num_actions, continuous=False, delta=1e-2, max_trajs=10):
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.states = []
        self.actions = []
        self.rewards = []
        self.traj_count = 0
        self.max_trajs = max_trajs
        self.delta = delta
        self.continuous = continuous
        
        self.x = tf.placeholder(tf.float32, shape=input_shape)
        self.r = tf.placeholder(tf.float32, shape=(None))
        self.a = tf.placeholder(tf.int32, shape=(None))
        self.idx = tf.one_hot(self.a, self.num_actions)
        self.actprob = self.build_network(layers)
        self.oldprob = self.build_network(layers, oldprob=1)
        
        t1 = tf.reduce_sum(self.actprob*self.idx, axis=1)
        t2 = tf.reduce_sum(self.oldprob*self.idx, axis=1)
        
        t = tf.exp(tf.log(t1+1e-7)-tf.log(t2+1e-7))
        self.L = tf.reduce_mean(t*self.r)
        self.kl = tf.reduce_mean(tf.reduce_sum(self.oldprob*(tf.log(self.oldprob)-tf.log(self.actprob)), axis=1))
        
        variable_lists = tf.concat(self.variable_placeholders, axis=0)
        variable_shape = variable_lists.get_shape()
        
        self.v = tf.placeholder(tf.float32, shape=(variable_shape))
        
        kl_grads = tf.gradients(self.kl, self.variable_placeholders)
        kl_grad = tf.concat(kl_grads, axis=0)
        kl_grads2 = tf.gradients(tf.reduce_sum(kl_grad*self.v), self.variable_placeholders)
        self.d2kl = tf.concat(kl_grads2, axis=0)
        
        L_grads = tf.gradients(self.L, self.variable_placeholders)
        self.L_grad = tf.concat(L_grads, axis=0)
        
        self.session = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4))
        init_op = tf.global_variables_initializer()

        self.session.run(init_op)
        self.copy_variables()
    
    def copy_variables(self):
        for i in range(len(self.old_variables)):
            self.old_variables[i] = self.variables[i]
    
    def construct_feeddict(self, feed_dict):
        for x, y in zip(self.variables, self.variable_placeholders):
            feed_dict[y] = x.reshape(-1, 1)
        
        for x, y in zip(self.old_variables, self.old_variable_placeholders):
            feed_dict[y] = x.reshape(-1, 1)
        
    def choose_action(self, s):
        feed_dict = {self.x : [s]} #, self.a : actions, self.r : rewards}
        self.construct_feeddict(feed_dict)
        
        p = self.session.run(self.actprob, feed_dict=feed_dict)
        
        try:
            a = np.random.choice(self.num_actions, size=1, p=p[0])
            return a
        except Warning:
            out = self.session.run(self.out, feed_dict=feed_dict)
            print(out)
            return 0
        
    def update(self):
        
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        
        feed_dict = {self.x : states, self.a : actions, self.r : rewards}
        self.construct_feeddict(feed_dict)
        
        g = self.session.run(self.L_grad, feed_dict=feed_dict)
        
        def f_Ax(p):
            feed_dict[self.x] = states[np.random.choice(len(states), size=min(len(states),50))] 
            feed_dict[self.a] = actions[np.random.choice(len(actions), size=min(len(states),50))] 
            feed_dict[self.v] = p
            val = self.session.run(self.d2kl, feed_dict=feed_dict) #+p*1e-3
            feed_dict[self.x] = states 
            feed_dict[self.a] = actions
            
            return val
            
        s_unscaled = self.cg(f_Ax, g)#self.conjugate_gradient(self.session, self.d2kl, self.v, g, feed_dict)
        
        s = np.sqrt(2*self.delta/(1e-8 + s_unscaled.T.dot(f_Ax(s_unscaled))))*s_unscaled
        alpha = 1
        
        obj = self.session.run(self.L, feed_dict=feed_dict)
        KL = self.session.run(self.kl, feed_dict=feed_dict)
        #print("obj" , obj)
        
        #do the line search
        while alpha > 1e-7:
            
            self.increment(s, alpha)
            
            self.construct_feeddict(feed_dict)
            
            L = self.session.run(self.L, feed_dict=feed_dict)
            KL = self.session.run(self.kl, feed_dict=feed_dict)
            
            #print("curr_val" , KL, L)
            
            if L > obj and KL < self.delta:
                break
            
            alpha *= 0.5
        
        #clean up
        self.states = []
        self.actions = []
        self.rewards = []
        self.traj_count = 0
        self.copy_variables()

    def increment(self, s, alpha):
        ind = 0
        for i in range(len(self.old_variables)):
            w = self.old_variables[i]
            self.variables[i] = w + alpha*s[ind:ind+len(w.reshape(-1))].reshape(w.shape)
            
            ind = ind+len(w.reshape(-1))
        
    def cg(self, f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
        """
        Demmel p 312
        """
        p = b.copy()
        r = b.copy()
        x = np.zeros_like(b)
        rdotr = r.T.dot(r)

        fmtstr =  "%10i %10.3g %10.3g"
        titlestr =  "%10s %10s %10s"
        if verbose: 
            print(titlestr % ("iter", "residual norm", "soln norm"))

        for i in range(cg_iters):
            if callback is not None:
                callback(x)
            if verbose: 
                print(fmtstr % (i, rdotr, np.linalg.norm(b-f_Ax(x))))
            z = f_Ax(p)
            v = rdotr / (p.T.dot(z) + 1e-8)
            x += v*p
            r -= v*z
            newrdotr = r.T.dot(r)
            mu = newrdotr / (rdotr + 1e-8)
            p = r + mu*p

            rdotr = newrdotr
            if rdotr < residual_tol:
                break

        if verbose: 
            print(fmtstr % (i+1, rdotr, np.linalg.norm(b-f_Ax(x))))  # pylint: disable=W0631
        return x
    
    def conjugate_gradient(self, f_Ax, g, iters=10, eps=0.01):
        
        n = len(g)
        x = np.zeros(n)
        
        r = g - f_Ax(x.reshape(-1,1))
        p = r
        k = 0
        while k < iters:
            alpha = np.sum(r*r)/np.dot(p.reshape(1,-1), f_Ax(p.reshape(-1,1)))
            
            x = x + alpha[0][0]*p.reshape(x.shape)
            
            r1 = r - alpha*f_Ax(p.reshape(-1,1))
            
            if np.max(np.abs(r1)) < eps:
                #print(np.sqrt(np.sum(r1*r1)), k)
                break
                
            beta = np.sum(r1*r1)/np.sum(r*r)
            r = r1
            p = r + beta*p
            k += 1
        
        print("diff " , np.linalg.norm(f_Ax(x.reshape(-1,1))-g), k)
        
        return x.reshape(-1, 1)
    
    def add_trajectory(self, states, actions, rewards):
        self.states += states.tolist()
        self.actions += actions.tolist()
        self.rewards += rewards.tolist()
        
        self.traj_count += 1
        
        if self.traj_count >= self.max_trajs:
            self.update()
    
    def build_network(self, layers, oldprob=0):
                
        out = self.x
        i = 0
        
        mu=0
        sigma=0.1
        variable_placeholders = []
        variables = []
            
        nw = "tw"
        nb = "tb"

        while i < len(layers):
            
            if layers[i] == 'conv':
                i += 1
                shape = layers[i]
                W = np.random.randn(*shape)*sigma + mu
                B = np.zeros(shape[-1])
                
                w = tf.placeholder(tf.float32, shape=shape)
                b = tf.placeholder(tf.float32, shape=shape[-1])
                out = tf.nn.conv2d(out, w, strides=[1,1,1,1], padding="VALID") + b
                out = tf.nn.relu(out)
                
                variables.append(W)
                variables.append(B)

                variable_placeholders.append(w)
                variable_placeholders.append(b)
            
            if layers[i] == 'flatten':
                out = flatten(out)
            
            if layers[i] == 'pool':
                out = tf.nn.max_pool(out, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID")
         
            if layers[i] == 'fc':
                i += 1
                din = out.get_shape().as_list()[1]
                dout = layers[i][1]
                
                W = np.random.randn(din, dout)*np.sqrt(2.0/din)
                B = np.zeros(dout)
                
                w = tf.placeholder(tf.float32, shape=(din*dout, 1))
                b = tf.placeholder(tf.float32, shape=(dout, 1))
                
                variables.append(W)
                variables.append(B)

                variable_placeholders.append(w)
                variable_placeholders.append(b)

                out = tf.matmul(out, tf.reshape(w, shape=(din, dout))) + tf.reshape(b, shape=[-1])
                out = tf.sigmoid(out)
                
            i += 1
        
        din = out.get_shape().as_list()[1]
        dout = self.num_actions
        
        W = np.random.randn(din, dout)*np.sqrt(2.0/din)
        B = np.zeros(dout)
        
        w = tf.placeholder(tf.float32, shape=(din*dout, 1))
        b = tf.placeholder(tf.float32, shape=(dout, 1))
        
        variables.append(W)
        variables.append(B)

        variable_placeholders.append(w)
        variable_placeholders.append(b)
        
        out = tf.matmul(out, tf.reshape(w, shape=(din, dout))) + tf.reshape(b, shape=[-1])
        
        probabilities = tf.nn.softmax(out)

        if oldprob == 0:
            self.variables = variables
            self.out = out
            self.variable_placeholders = variable_placeholders
        else:
            self.old_variables = variables
            self.old_variable_placeholders = variable_placeholders
            
        
        return probabilities