from collections import deque
import random
import numpy as np

class ReplayBuffer:
    
    def __init__(self, buffer_size):
        
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0
    
    def experience(self, s, a, r, sn):
        
        if np.sum(sn**2) == 0:
            f = 0
        else:
            f = 1
        
        exp = (s,a,r,sn,f)
        self.buffer.append(exp)
        self.count += 1
        
        if self.count > self.buffer_size:
            self.buffer.popleft()
            self.count -= 1
    
    def sample_minibatch(self, sample_size):
        
        l = self.count
        if l < sample_size:
            batch = random.sample(self.buffer, l)
        else:
            batch = random.sample(self.buffer, sample_size)
            
        s = np.array([_[0] for _ in batch])
        a = np.array([_[1] for _ in batch])
        r = np.array([_[2] for _ in batch])
        sn = np.array([_[3] for _ in batch])
        f = np.array([_[4] for _ in batch])
        
        return s, a, r, sn, f