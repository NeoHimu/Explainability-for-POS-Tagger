'''
@author: Leila Arras
@maintainer: Leila Arras
@date: 21.06.2017
@version: 1.0
@copyright: Copyright (c) 2017, Leila Arras, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license: BSD-2-Clause
'''

import numpy as np
import pickle
from numpy import newaxis as na
from code.LSTM.LRP_linear_layer import *


class LSTM_bidi:
    
    def __init__(self, model_path='./model/'):
        
        # vocabulary
        f_voc     = open(model_path + "vocab", 'rb')
        self.voc  = pickle.load(f_voc)
        f_voc.close()
        
        # word embeddings
        #self.E    = np.load(model_path + 'embeddings.npy', mmap_mode='r') # shape V*e
        
        # model weights
        f_model   = open(model_path + 'saved_model.pkl', 'rb')
        model     = pickle.load(f_model)
        f_model.close()
        
        # word embeddings
        self.E = model["embeddings"]
        # left encoder
        self.Wxh_Left  = model["Wxh_Left"]  # shape 4d*e
        #self.bxh_Left  = model["bxh_Left"]  # shape 4d 
        self.Whh_Left  = model["Whh_Left"]  # shape 4d*d
        self.bhh_Left  = model["bhh_Left"]  # shape 4d  
        # right encoder
        self.Wxh_Right = model["Wxh_Right"]
        #self.bxh_Right = model["bxh_Right"]
        self.Whh_Right = model["Whh_Right"]
        self.bhh_Right = model["bhh_Right"]   
        # softmax
        self.Why_Left  = model["Why_Left"]  # shape C*d
        self.Why_Right = model["Why_Right"] # shape C*d
    

    def set_input(self, w, delete_pos=None):
        """
        Build the numerical input x/x_rev from word sequence indices w (+ initialize hidden layers h, c)
        Optionally delete words at positions delete_pos.
        """
        T      = len(w)                         # input word sequence length
        d      = int(self.Wxh_Left.shape[0]/4)  # hidden layer dimension
        e      = self.E.shape[1]                # word embedding dimension
        x      = np.zeros((T, e))
        x[:,:] = self.E[w,:]
        if delete_pos is not None:
            x[delete_pos, :] = np.zeros((len(delete_pos), e))
        
        self.w              = w
        self.x              = x
        self.x_rev          = x[::-1,:].copy()
        
        self.h_Left         = np.zeros((T+1, d))
        self.c_Left         = np.zeros((T+1, d))
        self.h_Right        = np.zeros((T+1, d))
        self.c_Right        = np.zeros((T+1, d))
     
   
    def forward(self):
        """
        Update the hidden layer values (using model weights and numerical input x/x_rev previously built from word sequence w)
        """
        T      = len(self.w)                         
        d      = int(self.Wxh_Left.shape[0]/4)      
        idx    = np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) # indices of the gates i,f,o
          
        # initialize
        self.s = []# prediction scores
        
        self.gates_xh_Left  = np.zeros((T, 4*d))  
        self.gates_hh_Left  = np.zeros((T, 4*d)) 
        self.gates_pre_Left = np.zeros((T, 4*d))  # gates i, g, f, o pre-activation
        self.gates_Left     = np.zeros((T, 4*d))  # gates i, g, f, o activation
        
        self.gates_xh_Right = np.zeros((T, 4*d))  
        self.gates_hh_Right = np.zeros((T, 4*d)) 
        self.gates_pre_Right= np.zeros((T, 4*d))
        self.gates_Right    = np.zeros((T, 4*d)) 
             
        for t in range(T): 
            self.gates_xh_Left[t]    = np.dot(self.Wxh_Left, self.x[t])       # + self.bxh_Left
            self.gates_hh_Left[t]    = np.dot(self.Whh_Left, self.h_Left[t-1]) + self.bhh_Left
            self.gates_pre_Left[t]   = self.gates_xh_Left[t] + self.gates_hh_Left[t]
            self.gates_Left[t,idx]   = 1.0/(1.0 + np.exp(- self.gates_pre_Left[t,idx]))
            self.gates_Left[t,d:2*d] = np.tanh(self.gates_pre_Left[t,d:2*d]) 
            self.c_Left[t]           = self.gates_Left[t,2*d:3*d]*self.c_Left[t-1] + self.gates_Left[t,0:d]*self.gates_Left[t,d:2*d]
            self.h_Left[t]           = self.gates_Left[t,3*d:4*d]*np.tanh(self.c_Left[t])
            
            self.gates_xh_Right[t]    = np.dot(self.Wxh_Right, self.x_rev[t])    # + self.bxh_Right
            self.gates_hh_Right[t]    = np.dot(self.Whh_Right, self.h_Right[t-1]) + self.bhh_Right
            self.gates_pre_Right[t]   = self.gates_xh_Right[t] + self.gates_hh_Right[t]
            self.gates_Right[t,idx]   = 1.0/(1.0 + np.exp(- self.gates_pre_Right[t,idx]))
            self.gates_Right[t,d:2*d] = np.tanh(self.gates_pre_Right[t,d:2*d])                 
            self.c_Right[t]           = self.gates_Right[t,2*d:3*d]*self.c_Right[t-1] + self.gates_Right[t,0:d]*self.gates_Right[t,d:2*d]
            self.h_Right[t]           = self.gates_Right[t,3*d:4*d]*np.tanh(self.c_Right[t])
            
        for t in range(T):
            self.y_Left  = np.dot(self.Why_Left,  self.h_Left[t])
            self.y_Right = np.dot(self.Why_Right, self.h_Right[t])
            self.s.append(self.y_Left + self.y_Right)
        
        return self.s.copy() # prediction scores
     
        
        
    def backward(self, w, word_position, sensitivity_class):
        """
        Update the hidden layer gradients by backpropagating a gradient of 1.0 for the class sensitivity_class
        """
        # forward pass
        self.set_input(w)
        self.forward() 
        
        T      = len(self.w)
        d      = int(self.Wxh_Left.shape[0]/4)
        C      = self.Why_Left.shape[0]   # number of classes
        idx    = np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) 
        
        # initialize
        self.dx               = np.zeros(self.x.shape)
        self.dx_rev           = np.zeros(self.x.shape)
        
        self.dh_Left          = np.zeros((T+1, d))
        self.dc_Left          = np.zeros((T+1, d))
        self.dgates_pre_Left  = np.zeros((T, 4*d))  # gates i, g, f, o pre-activation
        self.dgates_Left      = np.zeros((T, 4*d))  # gates i, g, f, o activation
        
        self.dh_Right         = np.zeros((T+1, d))
        self.dc_Right         = np.zeros((T+1, d))
        self.dgates_pre_Right = np.zeros((T, 4*d))  # gates i, g, f, o pre-activation
        self.dgates_Right     = np.zeros((T, 4*d))  # gates i, g, f, o activation
               
        ds                    = np.zeros((C))
        ds[sensitivity_class] = 1.0   # * np.exp(self.s[sensitivity_class]) # add this term to reproduce results from (Li et al., 2016)
        dy_Left               = ds.copy()
        dy_Right              = ds.copy()
        
        self.dh_Left[word_position-1]     = np.dot(self.Why_Left.T,  dy_Left)
        self.dh_Right[T-word_position-1]    = np.dot(self.Why_Right.T, dy_Right)
        
        for t in reversed(range(word_position)): 
            self.dgates_Left[t,3*d:4*d]  = self.dh_Left[t] * np.tanh(self.c_Left[t])     # do[t]
            self.dc_Left[t]             += self.dh_Left[t] * self.gates_Left[t,3*d:4*d] * (1.-(np.tanh(self.c_Left[t]))**2) # dc[t]
            self.dgates_Left[t,2*d:3*d]  = self.dc_Left[t] * self.c_Left[t-1]            # df[t]
            self.dc_Left[t-1]            = self.dc_Left[t] * self.gates_Left[t,2*d:3*d]  # dc[t-1]
            self.dgates_Left[t,0:d]      = self.dc_Left[t] * self.gates_Left[t,d:2*d]    # di[t]
            self.dgates_Left[t,d:2*d]    = self.dc_Left[t] * self.gates_Left[t,0:d]      # dg[t]
            self.dgates_pre_Left[t,idx]  = self.dgates_Left[t,idx] * self.gates_Left[t,idx] * (1.0 - self.gates_Left[t,idx]) # d ifo pre[t]
            self.dgates_pre_Left[t,d:2*d]= self.dgates_Left[t,d:2*d] *  (1.-(self.gates_Left[t,d:2*d])**2) # d g pre[t]
            self.dh_Left[t-1]            = np.dot(self.Whh_Left.T, self.dgates_pre_Left[t])
            self.dx[t]                   = np.dot(self.Wxh_Left.T, self.dgates_pre_Left[t])
            
            
        for t in reversed(range(T-word_position)): 
            self.dgates_Right[t,3*d:4*d]  = self.dh_Right[t] * np.tanh(self.c_Right[t])         
            self.dc_Right[t]             += self.dh_Right[t] * self.gates_Right[t,3*d:4*d] * (1.-(np.tanh(self.c_Right[t]))**2) 
            self.dgates_Right[t,2*d:3*d]  = self.dc_Right[t] * self.c_Right[t-1]            
            self.dc_Right[t-1]            = self.dc_Right[t] * self.gates_Right[t,2*d:3*d] 
            self.dgates_Right[t,0:d]      = self.dc_Right[t] * self.gates_Right[t,d:2*d]    
            self.dgates_Right[t,d:2*d]    = self.dc_Right[t] * self.gates_Right[t,0:d]      
            self.dgates_pre_Right[t,idx]  = self.dgates_Right[t,idx] * self.gates_Right[t,idx] * (1.0 - self.gates_Right[t,idx]) 
            self.dgates_pre_Right[t,d:2*d]= self.dgates_Right[t,d:2*d] *  (1.-(self.gates_Right[t,d:2*d])**2) 
            self.dh_Right[t-1]            = np.dot(self.Whh_Right.T, self.dgates_pre_Right[t])
            self.dx_rev[t]                = np.dot(self.Wxh_Right.T, self.dgates_pre_Right[t])
                    
        return self.dx.copy(), self.dx_rev[::-1,:].copy()     
            
        
    
                   
    def lrp(self, w, word_position, LRP_class, eps=0.001, bias_factor=1.0):#word_position starts from 1
        """
        Update the hidden layer relevances by performing LRP for the target class LRP_class
        """
        # forward pass
        self.set_input(w)
        self.forward() 
        
        T      = len(self.w)
        d      = int(self.Wxh_Left.shape[0]/4)
        e      = self.E.shape[1] 
        C      = self.Why_Left.shape[0]  # number of classes
        idx    = np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) 
        
        # initialize
        Rx       = np.zeros(self.x.shape)
        Rx_rev   = np.zeros(self.x.shape)
        
        Rh_Left  = np.zeros((T+1, d))
        Rc_Left  = np.zeros((T+1, d))
        Rg_Left  = np.zeros((T,   d)) # gate g only
        Rh_Right = np.zeros((T+1, d))
        Rc_Right = np.zeros((T+1, d))
        Rg_Right = np.zeros((T,   d)) # gate g only
        
        Rout_mask            = np.zeros((C))
        Rout_mask[LRP_class] = 1.0  
        #print(Rh_Left[word_position-1])
        # format reminder: lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        print(self.s[word_position-1].shape)
        print((self.s[word_position-1]*Rout_mask).shape)
        Rh_Left[word_position-1]  = lrp_linear(self.h_Left[word_position-1],  
                                               self.Why_Left.T , 
                                               np.zeros((C)), 
                                               self.s[word_position-1], 
                                               self.s[word_position-1]*Rout_mask, 
                                               2*d, 
                                               eps, 
                                               bias_factor, 
                                               debug=False)
        
        Rh_Right[T-word_position-1] = lrp_linear(self.h_Right[word_position-1], 
                                                 self.Why_Right.T, 
                                                 np.zeros((C)), 
                                                 self.s[word_position-1], 
                                                 self.s[word_position-1]*Rout_mask, 
                                                 2*d, 
                                                 eps, 
                                                 bias_factor, 
                                                 debug=False)
        
        for t in reversed(range(word_position)):
            Rc_Left[t]   += Rh_Left[t]
            Rc_Left[t-1]  = lrp_linear(self.gates_Left[t,2*d:3*d]*self.c_Left[t-1],     np.identity(d), np.zeros((d)), self.c_Left[t], Rc_Left[t], 2*d, eps, bias_factor, debug=False)
            Rg_Left[t]    = lrp_linear(self.gates_Left[t,0:d]*self.gates_Left[t,d:2*d], np.identity(d), np.zeros((d)), self.c_Left[t], Rc_Left[t], 2*d, eps, bias_factor, debug=False)
            Rx[t]         = lrp_linear(self.x[t],        self.Wxh_Left[d:2*d].T, self.bhh_Left[d:2*d], self.gates_pre_Left[t,d:2*d], Rg_Left[t], d+e, eps, bias_factor, debug=False)
            Rh_Left[t-1]  = lrp_linear(self.h_Left[t-1], self.Whh_Left[d:2*d].T, self.bhh_Left[d:2*d], self.gates_pre_Left[t,d:2*d], Rg_Left[t], d+e, eps, bias_factor, debug=False)
            
        for t in reversed(range(T-word_position)):
            Rc_Right[t]  += Rh_Right[t]
            Rc_Right[t-1] = lrp_linear(self.gates_Right[t,2*d:3*d]*self.c_Right[t-1],     np.identity(d), np.zeros((d)), self.c_Right[t], Rc_Right[t], 2*d, eps, bias_factor, debug=False)
            Rg_Right[t]   = lrp_linear(self.gates_Right[t,0:d]*self.gates_Right[t,d:2*d], np.identity(d), np.zeros((d)), self.c_Right[t], Rc_Right[t], 2*d, eps, bias_factor, debug=False)
            Rx_rev[t]     = lrp_linear(self.x_rev[t],     self.Wxh_Right[d:2*d].T, self.bhh_Right[d:2*d], self.gates_pre_Right[t,d:2*d], Rg_Right[t], d+e, eps, bias_factor, debug=False)
            Rh_Right[t-1] = lrp_linear(self.h_Right[t-1], self.Whh_Right[d:2*d].T, self.bhh_Right[d:2*d], self.gates_pre_Right[t,d:2*d], Rg_Right[t], d+e, eps, bias_factor, debug=False)
                   
        return Rx, Rx_rev[::-1,:], Rh_Left[-1].sum()+Rc_Left[-1].sum()+Rh_Right[-1].sum()+Rc_Right[-1].sum()

