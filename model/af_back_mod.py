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


def self_sup(x,batch_size,length,ss_hide,key=None):
    if ss_hide == 1 or key is None:
        return x,x
    else:
        tmp = jax.random.uniform(key,[batch_size,length,1]) # [128,52,1]
        #mask = (tmp > ss_hide).astype(x.dtype) # [128,52,1]
        mask = (tmp > ss_hide).astype('float32')
    return x*mask, x*(1-mask) # x = X[index] [128,52,20]

# get alignment
def get_aln(z, lengths,sw_unroll,gap,open,temp,key=None): 
    # local-alignment (smith-waterman)
    if gap is None:
        aln_app = sw.sw_nogap(batch=True, unroll=sw_unroll)
        aln = aln_app(z, lengths, temp)
    elif open is None:
        aln_app = sw.sw(batch=True, unroll=sw_unroll)
        aln = aln_app(z, lengths, gap, temp)
    else:
        aln_app = sw.sw_affine(restrict_turns=True, batch=True, unroll=sw_unroll)
        aln = aln_app(z, lengths, gap, open, temp)
    return aln


############################################
def mod(msa_params, key, inputs, model_params, af2_inputs, model_runner, supervised=None, unsupervised=None, confidence=None):

  # get embedding per sequence
  emb = laxy.Conv1D(msa_params["emb"])(af2_inputs["x"])

  # get similarity matrix
  lengths = jnp.stack([af2_inputs["lengths"], jnp.broadcast_to(af2_inputs['lengths'][0],af2_inputs["lengths"].shape)],-1)
  sm_mtx = emb @ emb[0,:af2_inputs['lengths'][0]].T
  sm_mask = jnp.broadcast_to(af2_inputs["x"].sum(-1,keepdims=True), sm_mtx.shape)
  sm_mtx = nf.norm_row_col(sm_mtx, sm_mask, norm_mode="fast")

  # get alignment
  aln = sw.sw()(sm_mtx, lengths, msa_params["gap"], msa_params["temp"]) # use param['temp']

  # get msa
  x_msa = jnp.einsum("...ia,...ij->...ja", af2_inputs["x"], aln)
  x_msa = x_msa.at[0,:,:].set(af2_inputs["x"][0,:af2_inputs['lengths'][0],:])

  # updata mrf param
  x_msa_pred = laxy.MRF(msa_params["mrf"])(x_msa, return_w=False)
  x_msa_pred = jax.nn.softmax(x_msa_pred, -1)

  # add gap character
  x_gap = jax.nn.relu(1 - x_msa.sum(-1,keepdims=True))
  #x_msa_gap = jnp.concatenate([x_msa,jnp.zeros_like(x_gap),x_gap],-1)
  x_msa_gap = jnp.concatenate([x_msa_pred,jnp.zeros_like(x_gap),x_gap],-1)

  # update msa
  inputs_mod = inputs
  inputs_mod["msa_feat"] = jnp.zeros_like(inputs["msa_feat"]).at[...,0:22].set(x_msa_gap).at[...,25:47].set(x_msa_gap)

  # get alphafold outputs
  outputs = model_runner.apply(model_params, key, inputs_mod)

  #################
  # compute loss
  #################
  # distance to correct solution

  rmsd_loss = jnp_rmsd(af2_inputs["protein_obj"].atom_positions[:,1,:],
                       outputs["structure_module"]["final_atom_positions"][:,1,:])

  loss = 0
  losses = {"rmsd":rmsd_loss}

  if supervised:
    dgram_loss = get_dgram_loss(af2_inputs['batch'], outputs, model_config=model_runner.config)
    fape_loss = get_fape_loss(af2_inputs['batch'], outputs, model_config=model_runner.config)
    loss += dgram_loss + fape_loss
    losses.update({"dgram":dgram_loss, "fape":fape_loss})

  if unsupervised:
    x_msa_pred_logits = outputs["masked_msa"]["logits"]
    x_ms_pred_logits = jnp.einsum("...ja,...ij->...ia", x_msa_pred_logits, aln)
    x_ms_pred_log_softmax = jax.nn.log_softmax(x_ms_pred_logits[...,:22])[...,:20]
    cce_loss = -(af2_inputs["x"] * x_ms_pred_log_softmax).sum() / af2_inputs["x"].sum()
    loss += cce_loss
    losses.update({"cce":cce_loss})
      
  if confidence:
    pae_loss = jax.nn.softmax(outputs["predicted_aligned_error"]["logits"])
    pae_loss = (pae_loss * jnp.arange(pae_loss.shape[-1])).sum(-1).mean()
    plddt_loss = jax.nn.softmax(outputs["predicted_lddt"]["logits"])
    plddt_loss = (plddt_loss * jnp.arange(plddt_loss.shape[-1])[::-1]).sum(-1).mean()
    loss += pae_loss + plddt_loss
    losses.update({"pae":pae_loss, "plddt":plddt_loss})
      
  outs = {"final_atom_positions":outputs["structure_module"]["final_atom_positions"],
          "final_atom_mask":outputs["structure_module"]["final_atom_mask"]}

  return loss, ({"plddt": get_plddt(outputs),
                 "losses":losses, "outputs":outs,                   
                 "msa":x_msa_gap, "seq":x_msa[0]})




######################################

def mod1(msa_params, key, inputs, model_params, af2_inputs, model_runner, supervised=None, unsupervised=None, confidence=None):

  # get embedding per sequence
  x_ms_in, x_ms_out = self_sup(af2_inputs["x"],128,af2_inputs["x"].shape[1],0.15,key)
  x_ms_in_ = jnp.concatenate([af2_inputs["x"][:1], x_ms_in],0)
  #emb = nf.Conv1D_custom(msa_params["emb"])(x_ms_in_,key,scale=0.1)
  emb = laxy.Conv1D(msa_params["emb"])(x_ms_in_)

  # get similarity matrix
  sm_mtx = emb[1:] @ emb[0,:af2_inputs['lengths'][0]].T
  sm_mask = jnp.broadcast_to(af2_inputs["x"].sum(-1,keepdims=True), sm_mtx.shape)
  lengths = jnp.stack([af2_inputs["lengths"],jnp.broadcast_to(af2_inputs["lengths"][0],af2_inputs["lengths"].shape)],-1)
  sm_mtx = nf.norm_row_col(sm_mtx, sm_mask, norm_mode="fast")

  # get alignment
  aln = sw.sw()(sm_mtx, lengths, msa_params["gap"], msa_params["temp"]) # use param['temp']
  x_msa = jnp.einsum("...ia,...ij->...ja", x_ms_in, aln)
  x_msa = x_msa.at[0,:,:].set(af2_inputs["x"][0,:af2_inputs['lengths'][0],:])

  # updata mrf param
  x_msa_pred = laxy.MRF(msa_params["mrf"])(x_msa, return_w=False)
  x_msa_pred = jax.nn.softmax(x_msa_pred, -1)

  # add gap character
  x_gap = jax.nn.relu(1 - x_msa.sum(-1,keepdims=True))
  x_msa_gap = jnp.concatenate([x_msa_pred,jnp.zeros_like(x_gap),x_gap],-1)

  # update msa
  inputs_mod = inputs
  inputs_mod["msa_feat"] = jnp.zeros_like(inputs["msa_feat"]).at[...,0:22].set(x_msa_gap).at[...,25:47].set(x_msa_gap)

  # get alphafold outputs
  outputs = model_runner.apply(model_params, key, inputs_mod)

  #################
  # compute loss
  #################
  # distance to correct solution

  rmsd_loss = jnp_rmsd(af2_inputs["protein_obj"].atom_positions[:,1,:],
                       outputs["structure_module"]["final_atom_positions"][:,1,:])

  loss = 0
  losses = {"rmsd":rmsd_loss}

  if supervised:
    dgram_loss = get_dgram_loss(af2_inputs['batch'], outputs, model_config=model_runner.config)
    fape_loss = get_fape_loss(af2_inputs['batch'], outputs, model_config=model_runner.config)
    loss += dgram_loss + fape_loss
    losses.update({"dgram":dgram_loss, "fape":fape_loss})

  if unsupervised:
    x_msa_pred_logits = outputs["masked_msa"]["logits"]
    x_ms_pred_logits = jnp.einsum("...ja,...ij->...ia", x_msa_pred_logits, aln)
    x_ms_pred_log_softmax = jax.nn.log_softmax(x_ms_pred_logits[...,:22])[...,:20]
    cce_loss = -(af2_inputs["x"] * x_ms_pred_log_softmax).sum() / af2_inputs["x"].sum()
    loss += cce_loss
    losses.update({"cce":cce_loss})
      
  if confidence:
    pae_loss = jax.nn.softmax(outputs["predicted_aligned_error"]["logits"])
    pae_loss = (pae_loss * jnp.arange(pae_loss.shape[-1])).sum(-1).mean()
    plddt_loss = jax.nn.softmax(outputs["predicted_lddt"]["logits"])
    plddt_loss = (plddt_loss * jnp.arange(plddt_loss.shape[-1])[::-1]).sum(-1).mean()
    loss += pae_loss + plddt_loss
    losses.update({"pae":pae_loss, "plddt":plddt_loss})
      
  outs = {"final_atom_positions":outputs["structure_module"]["final_atom_positions"],
          "final_atom_mask":outputs["structure_module"]["final_atom_mask"]}

  return loss, ({"plddt": get_plddt(outputs),
                 "losses":losses, "outputs":outs,                   
                 "msa":x_msa_gap, "seq":x_msa[0]})

