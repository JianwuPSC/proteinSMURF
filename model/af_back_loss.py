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

def get_feat_idx(filename,idx,alphabet="ARNDCQEGHILKMFPSTWYV"):
  '''
  Given A3M file (from hhblits)
  return MSA (aligned), MS (unaligned) and ALN (alignment)
  '''
  def parse_fasta(filename):
    '''function to parse fasta file'''    
    header, sequence,new_header,new_seq = [],[],[],[]
    lines = open(filename, "r")
    seen_header=set()
    seen_seq=set()
    for line in lines:
      line = line.rstrip()
      if len(line) == 0: pass
      else:
        if line[0] == ">":
          header.append(line[1:])
          sequence.append([])
        else:
          sequence[-1].append(line)
    lines.close()
    sequence = [''.join(seq) for seq in sequence]

    for index,seq in enumerate(sequence):
        if seq in seen_seq:
            continue
        seen_seq.add(seq)
        new_seq.append(seq)
        
    for index,head in enumerate(header):
        if head in seen_header:
            continue
        seen_header.add(head)
        new_header.append(head)
 
    return new_header, new_seq

  names, seqs = parse_fasta(filename)

  names=list(np.array(names)[idx])
  seqs=list(np.array(seqs)[idx])

  a2n = {a:n for n,a in enumerate(alphabet)}
  def get_seqref(x):
    n,seq,ref,aligned_seq = 0,[],[],[]
    for aa in list(x):
      if aa != "-":
        seq.append(a2n.get(aa.upper(),-1))
        if aa.islower(): ref.append(-1); n -= 1
        else: ref.append(n); aligned_seq.append(seq[-1])
      else: aligned_seq.append(-1)
      n += 1
    return seq, ref, aligned_seq
  
  # get the multiple sequence alignment
  max_len = 0
  ms, aln, msa = [],[],[]
  for seq in seqs:
    seq_,ref_,aligned_seq_ = get_seqref(seq)
    if len(seq_) > max_len: max_len = len(seq_)
    ms.append(seq_)
    msa.append(aligned_seq_)
    aln.append(ref_)
  
  return msa, ms, aln


def prep_inputs_idx(a3m_file,pdb_file,idx):

  _, ms, aln = get_feat_idx(a3m_file,idx)
  lens = np.asarray([len(m) for m in ms])
  ms = nf.one_hot(nf.pad_max(ms))
  aln = nf.one_hot(nf.pad_max(aln))
  N = len(ms)
  protein_obj = protein.from_pdb_string(pdb_to_string(pdb_file))
  batch = {'aatype': protein_obj.aatype,
           'all_atom_positions': protein_obj.atom_positions,
           'all_atom_mask': protein_obj.atom_mask}
  batch.update(all_atom.atom37_to_frames(**batch)) # for fape calculcation
  msa, mtx = parsers.parse_a3m(open(a3m_file,"r").read())
  feature_dict = {
      **pipeline.make_sequence_features(sequence=msa[0],description="none",num_res=len(msa[0])),
      **pipeline.make_msa_features(msas=[msa], deletion_matrices=[mtx])
  }
  feature_dict["residue_index"] = protein_obj.residue_index
  return {"N":N,"lens":lens,
          "ms":ms,"aln":aln,
          "feature_dict":feature_dict,
          "protein_obj":protein_obj, "batch":batch}

##################################################


def get_model_runner(num_seq, model_name="model_3_ptm", dropout=False, backprop_recycles=False):
  # setup which model params to use
  model_config = config.model_config(model_name)
  model_config.model.global_config.use_remat = True

  model_config.model.num_recycle = 3
  model_config.data.common.num_recycle = 3

  model_config.data.eval.max_msa_clusters = num_seq
  model_config.data.common.max_extra_msa = 1
  model_config.data.eval.masked_msa_replace_fraction = 0

  # backprop through recycles
  model_config.model.backprop_recycle = backprop_recycles
  model_config.model.embeddings_and_evoformer.backprop_dgram = backprop_recycles

  if not dropout:
    model_config = set_dropout(model_config,0)

  # setup model
  model_params = get_model_haiku_params(model_name=model_name, data_dir="/home/wuj/data/tools/AF2/AF2/Reduced_dbs/")
  model_runner = model.RunModel(model_config, model_params, is_training=True)
  return model_runner, model_params

#############################################

def get_grad_fn(model_runner, x_ref_len, confidence=True, supervised=False, unsupervised=False, batch=None):
  def mod(msa_params, key, inputs, model_params, msa_inputs):

    # get embedding per sequence
    emb = laxy.Conv1D(msa_params["emb"])(msa_inputs["x"])

    # get similarity matrix
    lengths = jnp.stack([msa_inputs["lengths"], jnp.broadcast_to(x_ref_len,msa_inputs["lengths"].shape)],-1)
    sm_mtx = emb @ emb[0,:x_ref_len].T
    sm_mask = jnp.broadcast_to(msa_inputs["x"].sum(-1,keepdims=True), sm_mtx.shape)
    sm_mtx = nf.norm_row_col(sm_mtx, sm_mask, norm_mode="fast")

    # get alignment
    aln = sw.sw()(sm_mtx, lengths, msa_params["gap"], 1.0)

    # get msa
    x_msa = jnp.einsum("...ia,...ij->...ja", msa_inputs["x"], aln)
    x_msa = x_msa.at[0,:,:].set(msa_inputs["x"][0,:x_ref_len,:])
    
    # add gap character
    x_gap = jax.nn.relu(1 - x_msa.sum(-1,keepdims=True))
    x_msa_gap = jnp.concatenate([x_msa,jnp.zeros_like(x_gap),x_gap],-1)
#    x_msa_gap1 = x_msa_gap[None,...]
#    x_msa_gap = repeat(x_msa_gap1,'1 b l c -> 4 b l c')

    # update msa
    inputs_mod = inputs
    inputs_mod["msa_feat"] = jnp.zeros_like(inputs["msa_feat"]).at[...,0:22].set(x_msa_gap).at[...,25:47].set(x_msa_gap)

    # get alphafold outputs
    outputs = model_runner.apply(model_params, key, inputs_mod)

    #################
    # compute loss
    #################
    # distance to correct solution
#    rmsd_loss = jnp_rmsd(INPUTS["protein_obj"].atom_positions[:,1,:],
#                         outputs["structure_module"]["final_atom_positions"][:,1,:])

    rmsd_loss = jnp_rmsd(INPUTS_idx["protein_obj"].atom_positions[:,1,:],
                         outputs["structure_module"]["final_atom_positions"][:,1,:])

    loss = 0
    losses = {"rmsd":rmsd_loss}
    if supervised:
      dgram_loss = get_dgram_loss(batch, outputs, model_config=model_runner.config)
      fape_loss = get_fape_loss(batch, outputs, model_config=model_runner.config)
      loss += dgram_loss + fape_loss
      losses.update({"dgram":dgram_loss, "fape":fape_loss})

    if unsupervised:
      x_msa_pred_logits = outputs["masked_msa"]["logits"]
      x_ms_pred_logits = jnp.einsum("...ja,...ij->...ia", x_msa_pred_logits, aln)
      x_ms_pred_log_softmax = jax.nn.log_softmax(x_ms_pred_logits[...,:22])[...,:20]
      cce_loss = -(msa_inputs["x"] * x_ms_pred_log_softmax).sum() / msa_inputs["x"].sum()
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
  
  return mod, jax.value_and_grad(mod, has_aux=True, argnums=0)

###############################


def get_grad_fn_nomsa(model_runner, x_ref_len, confidence=True, supervised=False, unsupervised=False, batch=None):
  def mod(msa_params, key, inputs, model_params, msa_inputs):

    # get embedding per sequence
    inputs_mod = inputs
    
    # get alphafold outputs
    outputs = model_runner.apply(model_params, key, inputs_mod)

    #################
    # compute loss
    #################
    # distance to correct solution
    rmsd_loss = jnp_rmsd(INPUTS["protein_obj"].atom_positions[:,1,:],
                         outputs["structure_module"]["final_atom_positions"][:,1,:])

    loss = 0
    losses = {"rmsd":rmsd_loss}
    if supervised:
      dgram_loss = get_dgram_loss(batch, outputs, model_config=model_runner.config)
      fape_loss = get_fape_loss(batch, outputs, model_config=model_runner.config)
      loss += dgram_loss + fape_loss
      losses.update({"dgram":dgram_loss, "fape":fape_loss})

    if unsupervised:
      x_msa_pred_logits = outputs["masked_msa"]["logits"]
      x_ms_pred_logits = jnp.einsum("...ja,...ij->...ia", x_msa_pred_logits, aln)
      x_ms_pred_log_softmax = jax.nn.log_softmax(x_ms_pred_logits[...,:22])[...,:20]
      cce_loss = -(msa_inputs["x"] * x_ms_pred_log_softmax).sum() / msa_inputs["x"].sum()
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
  
  return mod, jax.value_and_grad(mod, has_aux=True, argnums=0)
