import os
import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import adam

import numpy as np
import matplotlib.pyplot as plt

from einops import rearrange,reduce,repeat
#del utils
import sys

#sys.path.append('/home/wuj/data/tools/AF2/SMURF/af_backprop')
#sys.path.append('D:\\Data\\R_workstation\\deepmind\\SMURF_AFbackprop\\SMURF')
from utils import *

import laxy
import sw_functions as sw
import network_functions as nf

# import libraries

from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.model import data, config, model, features
from alphafold.data import parsers
from alphafold.model import all_atom
from alphafold.model.data import get_model_haiku_params

def Conv1D_custom(params=None):
  '''convolution'''
  def init_params(in_dims, out_dims, win, key):
    return {"w":jnp.zeros((out_dims,in_dims,win)),
            "b":jnp.zeros(out_dims)}  
  def layer(x, use_bias=True, stride=1, padding="SAME", key=None, scale=0.1):
    w = params["w"]
    if key is not None:
      w += scale * jax.random.normal(key, shape=w.shape)
    x = x.transpose([0,2,1])
    y = jax.lax.conv(x,w,(stride,),padding=padding)
    y = y.transpose([0,2,1]) 
    if use_bias: y += params["b"]
    return y
  if params is None: return init_params
  else: return layer

#############################################

def MRF(params=None):
  '''
  markov random field layer
  ----------------------------------------------------
  params = MRF()(L=length, A=alphabet, use_bias=True)
  output = MRF(params)(input, return_w=False)
  '''
  def init_params(L, A, use_bias=True, key=None, seed=None):
    params = {"w":jnp.zeros((L,A,L,A))}  # [52,20,52,20]
    if use_bias: params["b"] = jnp.zeros((L,A))  # [52,20]
    return params
  
  def layer(x, return_w=False, rm_diag=True, symm=True, mask=None):
    w = params["w"]
    L,A = w.shape[:2]
    if rm_diag:
      # set diagonal to zero
      w = w * (1-jnp.eye(L)[:,None,:,None]) # jnp.eye(L) 默认返回[L,L], [:,None,:None] 在0,3 列扩1维.
    if symm:
      # symmetrize
      w = 0.5 * (w + w.transpose([2,3,0,1])) # [0,1,2,3]+[2,3,0,1] 均匀化
    if mask is not None:
      w = w * mask[:,None,:,None]  # 二维 mask, 扩充至4维    
      
    y = jnp.tensordot(x,w,2) # x (N,L,A), w (L,A,L,A)
    if "b" in params: y += params["b"]
      
    if return_w: return y,w
    else: return y

  if params is None: return init_params
  else: return layer

##########################################

def smurf_params_init(X,lengths,lr,key,batch_size=128,win=18, filters=512,learn_bias=True,sw_open=None,sw_gap=-3.0,sw_temp=1.0):
    
    N,L,A = X.shape
    # inputs
    lengths = lengths
    x_ref = X[:1]
    x_ref_len = lengths[0]  # singal dim
    lr = lr*jnp.log(batch_size)/x_ref_len

    # seed for weight initialization and sub-sampling input    
    params = {"mrf": laxy.MRF()(x_ref_len, A, use_bias=learn_bias, key=key)} # w:[L,A,L,A]  b:[L,A]
    params["emb"] = nf.Conv1D_custom()(A,filters,win,key=key) # [512,20,18]
    params["open"] = sw_open
    params["gap"] = sw_gap
    params["temp"] = sw_temp
    params["msa"] = X[0,:x_ref_len,...]
    
    return params

#########################################

def tensor_equal(a, b):
    # 判断类型是否均为tensor
    if type(a) != type(b):
        return False
    # 判断形状相等
    if a.shape != b.shape:
        return False
    # 逐值对比后若有False则不相等
    if not tf.reduce_min(tf.cast(a == b, dtype=tf.float32)):
        return False
    return True
