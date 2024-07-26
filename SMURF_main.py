import argparse
import os
from pathlib import Path
import jax
import jax.numpy as jnp
from jax.example_libraries.optimizers import adam
import pickle

import numpy as np
import matplotlib.pyplot as plt

import sys
import importlib

import laxy
import sw_functions as sw
import network_functions as nf

sys.path.append('/home/wuj/data/tools/AF2/SMURF/SMURF')
sys.path.append('/home/wuj/data/tools/AF2/SMURF/af_backprop')
from utils import *

# import libraries
from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.model import data, config, model
from alphafold.data import parsers
from alphafold.model import all_atom

#importlib.reload(input_pipeline)
####
from model import build_hhsearch
from model import build_feature
from model import af_back_loss
from model import af_back_mod
from model import param_init_updata
from model import pseudo_seq
from model import msa_to_fasta

#conda activate AF2
#python ../SMURF_main.py --input_seq zz_CNR_sequence.fa --hhsearch_out train_hhsearch --pdb_out train_pdb --batch_size 128 --epochs 100

def get_args_parser():
    parser = argparse.ArgumentParser('SMURF-AF2_backprop protein', add_help=True)

    parser.add_argument("--input_seq", required=True, type=str, help="input signal sequence .fasta")
    parser.add_argument("--hhsearch_out", required=True, type=str, help="hhsearch outdir includes a3m and hhr")
    parser.add_argument("--pdb_out", required=True, type=str, help="pdb outdir include templ pdb and predict pdb")
    
    parser.add_argument("--test_seq", type=str,default=None, help="test seuence .fasta")
    parser.add_argument("--test_hhsearch_out", type=str,default=None, help=" test seuence .fasta")
    parser.add_argument("--test_pdb_output", type=str, default=None, help="test pdb output")
    
    parser.add_argument("--pred_seq", type=str,default=None, help="test seuence .fasta")
    parser.add_argument("--pred_hhsearch_out", type=str,default=None, help=" test seuence .fasta")
    parser.add_argument("--pred_pdb_output", type=str, default=None, help="test pdb output")

    parser.add_argument("--batch_size", type=int, default=128, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--seed", type=int, default=142, help="dataloader worker size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--af2_input_mask",required=True, type=bool,default=None, help="AF2 input feature mask or not; if mask use params fill up them")

    parser.add_argument("--confidence", type=bool, default=True, help="get_grad_fn confidence")
    parser.add_argument("--unsupervised", type=bool, default=True, help="get_grad_fn unsupervised")
    parser.add_argument("--supervised", type=bool, default=True, help="unsupervised supervised")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=0, help="CUDA device ids")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

    parser.add_argument("--dropout", type=bool, default=False, help="get_model_runner AF2 dropout")
    parser.add_argument("--backprop_recycles", type=bool, default=False, help="get_model_runner AF2 backwod_recycles")

    return parser


def main (args, mode = 'train'):

    if mode == 'train':

        print('====================================Training')

        init_fun, update_fun, get_params = adam(step_size=args.lr)

        file1 = open(f'{args.pdb_out}/pseudo_loss.txt', mode = 'a+')

    # hhsearch seq prog
        #hhsearch = build_hhsearch.Hhsearch(args.input_seq, args.hhsearch_out)
        #hhresult = hhsearch.call()

    # hhsearch feature input
        #hhsearch_feature = build_feature.Hhsearch_feature(args.input_seq, a3m_path=f'{args.hhsearch_out}/msa_total.a3m',hhr_path=f'{args.hhsearch_out}/pdb_hits.hhr', 
        #                                    pkl_out=f'{args.hhsearch_out}/dict_file.pkl')
        #feature = hhsearch_feature.call()
        #print(f"{feature['msa'].shape}  hhsearch in dataset")

    # prep inputs
        INPUTS = af_back_loss.prep_inputs_idx(f'{args.hhsearch_out}/msa_total.a3m', f'{args.pdb_out}/template_rank.pdb',idx=...)
        print(f"{INPUTS['N']}  sequence used to train and {args.batch_size} in a batch")

    # splite input train
 
        key = laxy.KEY(args.seed)

        for k in range(args.epochs):
            
            idx = np.random.randint(0,INPUTS['N'],size=args.batch_size)
            new_idx=np.delete(idx,len(idx)-1)
            new_idx=(np.insert(new_idx, 0, 0, axis=None))

            INPUTS_idx = af_back_loss.prep_inputs_idx(f'{args.hhsearch_out}/msa_total.a3m', f'{args.pdb_out}/template_rank.pdb', idx=idx)
            INPUTS_idx_af2 = af_back_loss.prep_inputs_idx(f'{args.hhsearch_out}/msa_total.a3m', f'{args.pdb_out}/template_rank.pdb', idx=new_idx)

            smurf_inputs = {"x":INPUTS_idx_af2["ms"], "lengths":INPUTS_idx_af2["lens"],"aln":INPUTS_idx_af2["aln"],
                            "batch":INPUTS_idx_af2['batch'],"x_ref":INPUTS_idx_af2["ms"][:1],'x_ref_len':INPUTS_idx_af2["lens"][:1], "key":key.get(10)}

            af2_inputs = {"x":INPUTS_idx_af2["ms"], "aln":INPUTS_idx_af2["aln"], "lengths":INPUTS_idx_af2["lens"],
                          "protein_obj":INPUTS_idx_af2['protein_obj'],"batch":INPUTS_idx_af2['batch']}

            #input_hh = {'aatype':feature['aatype'],'between_segment_residues':feature['between_segment_residues'],'domain_name':feature['domain_name'],
            #        'residue_index':feature['residue_index'],'seq_length':feature['seq_length'],'sequence':feature['sequence'],
            #        'deletion_matrix_int':feature['deletion_matrix_int'][idx],
            #        'msa':feature['msa'][idx],'num_alignments':feature['num_alignments'],'msa_species_identifiers':feature['msa_species_identifiers'][idx],
            #        'template_aatype':feature['template_aatype'],'template_all_atom_masks':feature['template_all_atom_masks'],
            #        'template_all_atom_positions':feature['template_all_atom_positions'],'template_domain_names':feature['template_domain_names'],
            #        'template_sequence':feature['template_sequence'],'template_sum_probs':feature['template_sum_probs']}

            #input_hh['num_alignments'][input_hh['num_alignments']==feature['num_alignments'][0]] = args.batch_size

            if k < 1.0 :
                init_params = param_init_updata.smurf_params_init(smurf_inputs["x"] ,smurf_inputs['lengths'], args.lr, smurf_inputs['key'][1])
                smurf_state = init_fun(init_params)
            else:
                smurf_state = smurf_state

            print(f'{get_params(smurf_state)["gap"]}')

    # SMURF model
            smurf_model = nf.MRF(X=smurf_inputs['x'], lengths=smurf_inputs['lengths'], batch_size=args.batch_size,
                                 sw_open=None, sw_gap=-3.0, sw_learn_gap=True, seed=args.seed)
    # SMURF model grade
            model = smurf_model._get_model(initialize_params=False,return_aln=False)
            fn_grad = jax.value_and_grad(lambda p,i: model(p,i)[1].sum())
            smurf_value, smurf_grad = fn_grad(get_params(smurf_state), smurf_inputs)
            smurf_state = update_fun(k, smurf_grad, smurf_state)
            emb_grad = param_init_updata.tensor_equal(get_params(smurf_state)['emb']['w'], smurf_grad['emb']['w'])
    # SMURF emb grade
            n=0
            while emb_grad :
                smurf_value, smurf_grad = fn_grad(get_params(smurf_state), smurf_inputs)
                smurf_state = update_fun(k, smurf_grad, smurf_state)
                emb_grad = param_init_updata.tensor_equal(get_params(smurf_state)['emb']['w'], smurf_grad['emb']['w'])
                n=n+1

            print(f'{n} {get_params(smurf_state)["gap"]}')
            x_ms_pred_1, cce_loss_pred_1 = model(get_params(smurf_state), smurf_inputs)
            
            #smurf_params = smurf_model.opt.get_params()
            #smurf_model.opt.set_params(smurf_params)

    # AF2 feature updata
            model_runner, model_params = af_back_loss.get_model_runner(INPUTS_idx_af2["N"], dropout=args.dropout, backprop_recycles=args.backprop_recycles)
            feat_inputs = model_runner.process_features(INPUTS_idx_af2['feature_dict'], random_seed=args.seed)

    # AF2 backword
            #loss_fn, grad_fn = af_back_loss.get_grad_fn(model_runner, x_ref_len=INPUTS["lens"][0],
            #                     confidence=args.confidence, unsupervised=args.unsupervised, supervised=args.supervised,
            #                     batch=INPUTS_idx["batch"])
            #grad_fn = jax.jit(grad_fn)
            
    # optimizer
            #init_fun, update_fun, get_params = adam(step_size=args.lr)
            #state = init_fun({"emb":smurf_params["emb"],"gap":smurf_params["gap"],"mrf":smurf_params["gap"]})
            #(loss, outs), grad = grad_fn(get_params(state), inputs['key'][0], feat_inputs, model_params, af2_inputs)

            if args.af2_input_mask is True:
                af2_loss, af2_outs = af_back_mod.mod1(get_params(smurf_state), smurf_inputs['key'][1], feat_inputs, model_params, af2_inputs, model_runner,
                                         supervised=args.supervised, unsupervised=args.unsupervised, confidence=args.confidence)
                (af2_loss_value,af2_outs_value),af2_grad = jax.value_and_grad(af_back_mod.mod1, has_aux=True, argnums=0)(get_params(smurf_state),smurf_inputs['key'][1],
                                                    feat_inputs, model_params, af2_inputs, model_runner,
                                                    supervised=args.supervised, unsupervised=args.unsupervised, confidence=args.confidence)
            else:
                af2_loss, af2_outs = af_back_mod.mod(get_params(smurf_state), smurf_inputs['key'][0], feat_inputs, model_params, af2_inputs, model_runner,
                                         supervised=args.supervised, unsupervised=args.unsupervised, confidence=args.confidence)
                (af2_loss_value,af2_outs_value),af2_grad = jax.value_and_grad(af_back_mod.mod, has_aux=True, argnums=0)(get_params(smurf_state),smurf_inputs['key'][0],
                                                    feat_inputs, model_params, af2_inputs, model_runner,
                                                    supervised=args.supervised, unsupervised=args.unsupervised, confidence=args.confidence)


            smurf_state = update_fun(k, af2_grad, smurf_state)

            losses = []
            losses.append([])
            plddt = af2_outs["plddt"].mean()            
            losses[-1].append([af2_loss,plddt])
            losses[-1][-1].append(af2_outs["losses"]["rmsd"])
            losses[-1][-1].append(af2_outs["losses"]["cce"])

            BEST_PLDDT = 0
    # save results
            if plddt > BEST_PLDDT:
                BEST_PLDDT = plddt

            print(f'=====model_train epoch:{k}\tsmurf_loss:{cce_loss_pred_1}\taf2_loss:{af2_loss}\tcce:{af2_outs["losses"]["cce"]}\tRMSD:{af2_outs["losses"]["rmsd"]}\tplddt:{plddt}')

           # with open(f"{args.pdb_out}/pseudo_loss.txt", 'w') as file1:
            print(f'{k}\tsmurf_loss:{cce_loss_pred_1}\taf2_loss:{af2_loss}\tcce:{af2_outs["losses"]["cce"]}\tRMSD:{af2_outs["losses"]["rmsd"]}\tplddt:{plddt}\n',file=file1)

            if k % 10 == 0:
                save_pdb(af2_outs,f"{args.pdb_out}/{k}_pred.pdb")

        file1.close()
        pickle.dump({"losses":losses,"lr":args.lr, "seed": args.seed, "param_smurf":get_params(smurf_state), "param_af2":model_params}, open(f"{args.pdb_out}/{args.epochs}_param.pkl","wb"))

###############################

    else:  #test
        print('================================ Testing (pseudo + params) pdb')
    
    # load params
    
        with open(f"{args.pdb_out}/{args.epochs}_param.pkl", 'rb') as f:
            loaded_params = pickle.load(f)

        loaded_variables = {"lr":loaded_params['lr'], "seed": loaded_params['seed'], "param_smurf":loaded_params['state'], "param_af2":loaded_params['model_params'],
                             "mode_sumrf":loaded_params['smurf_model'], "mode_af2":['grad_fn']}

    # generate batch_size  pseudo sequence
        pseudo_fa, true_fa = pseudo_seq.generate_random(args.input_seq, f'{args.test_seq}/pseudo_seq.fa', f'{args.test_seq}/true_seq.fa')
    # hhsearch 
        hhsearch = build_hhsearch.Hhsearch(pseudo_fa, args.test_hhsearch_out)
        hhresult = hhsearch.call()

    # hhsearch feature input
        hhsearch_feature = build_feature.Hhsearch_feature(pseudo_fa, a3m_path=f'{args.test_hhsearch_out}/msa_total.a3m',hhr_path=f'{args.test_hhsearch_out}/pdb_hits.hhr',
                                            pkl_out=f'{args.test_hhsearch_out}/dict_file.pkl')
        feature = hhsearch_feature.call()
         
        INPUTS = af_back_loss.prep_inputs_idx(f'{args.hhsearch_out}/msa_total.a3m', f'{args.pdb_out}/template_rank.pdb', idx=0)
        INPUTS_idx = af_back_loss.prep_inputs_idx(f'{args.test_hhsearch_out}/msa_total.a3m', f'{args.pdb_out}/template_rank.pdb', idx=0)
        INPUTS_idx_af2 = af_back_loss.prep_inputs_idx(f'{args.test_hhsearch_out}/msa_total.a3m', f'{args.pdb_out}/template_rank.pdb', idx=0)

        msa_inputs = {"x":INPUTS_idx_af2["ms"], "aln":INPUTS_idx_af2["aln"], "lengths":INPUTS_idx_af2["lens"]}

        input_hh = {'aatype':feature['aatype'],'between_segment_residues':feature['between_segment_residues'],'domain_name':feature['domain_name'],
                    'residue_index':feature['residue_index'],'seq_length':feature['seq_length'],'sequence':feature['sequence'],
                    'deletion_matrix_int':feature['deletion_matrix_int'],
                    'msa':feature['msa'],'num_alignments':feature['num_alignments'],'msa_species_identifiers':feature['msa_species_identifiers'],
                    'template_aatype':feature['template_aatype'],'template_all_atom_masks':feature['template_all_atom_masks'],
                    'template_all_atom_positions':feature['template_all_atom_positions'],'template_domain_names':feature['template_domain_names'],
                    'template_sequence':feature['template_sequence'],'template_sum_probs':feature['template_sum_probs']}

        inputs = {"x":INPUTS_idx["ms"], "lengths":INPUTS_idx["lens"],'aln':INPUTS_idx["aln"],
                  "x_ref":INPUTS["ms"][:1],'x_ref_len':INPUTS["lens"][:1], "key":key.get(10)}

    # SMURF model
        smurf_model = nf.MRF(X=inputs['x'], lengths=inputs['lengths'], batch_size=INPUTS['N'],
                              sw_open=None, sw_gap=-3.0, sw_learn_gap=True, seed=args.seed)
        smurf_params = smurf_model.opt.get_params()
        smurf_params['emb']=loaded_variables['param_smurf']['emb']
        smurf_params['mrf']=loaded_variables['param_smurf']['mrf']

        model_aln = jax.jit(smurf_model._get_model(initialize_params=False, return_aln=False))

        x_ms_pred, cce_loss_pred=model_aln(smurf_params,inputs)
    
    # x_ms_pred to fasta
        x_ms_pred=jax.nn.softmax(x_ms_pred, axis=-1)
        pre_fa = msa_to_fasta(x_ms_pred[0,:,:], args.pred_seq)

    # AF2 feature updata
        model_runner, model_params = af_back_loss.get_model_runner(INPUTS_idx["N"], dropout=args.dropout,backprop_recycles=args.backprop_recycles)

        feat_inputs = model_runner.process_features(input_hh, random_seed=args.seed)

    # AF2 backword
        loss_fn, grad_fn = af_back_loss.get_grad_fn(model_runner, x_ref_len=INPUTS["lens"][0],
                                 confidence=confidence, unsupervised=unsupervised, supervised=supervised,
                                 batch=INPUTS_idx["batch"])
        grad_fn = jax.jit(grad_fn)

        LOSSES = []
        BEST_PLDDT = 0
        losses = []
        loss = 0

    # optimizer
        init_fun, update_fun, get_params = adam(step_size=args.lr)

        state = loaded_variables['param_smurf']
        model_params = loaded_variables['param_af2']
        (loss, outs), grad = grad_fn(get_params(state), inputs['key'][0], feat_inputs, model_params, msa_inputs)
        plddt = outs["plddt"].mean()

        losses[-1].append([loss,plddt])
        losses[-1][-1].append(outs["losses"]["rmsd"])
        losses[-1][-1].append(outs["losses"]["cce"])

    # save results
        if plddt > BEST_PLDDT:
            BEST_PLDDT = plddt

        with open(f"{args.test_pdb_output}/pseudo_loss.txt", 'w') as file1:
            print(f'pseudo_model\tcce:{cce_loss_pred}\tRMSD:{outs["losses"]["rmsd"]}\tplddt:{plddt}',file=file1)
        file1.close()

        pickle.dump({"losses":losses}, open(f"{args.test_pdb_output}/pseudo_loss.pkl","wb"))

        save_pdb(outs,f"{args.test_pdb_output}/pseudo_model.pdb")
         

    # pred_fa hhsearch

        print('=================================== Testing (pseudo predict) pdb')

        hhsearch = build_hhsearch.Hhsearch(pre_fa, args.pred_hhsearch_out)
        hhresult = hhsearch.call()

        hhsearch_feature = build_feature.Hhsearch_feature(pre_fa, a3m_path=f'{args.pred_hhsearch_out}/msa_total.a3m',hhr_path=f'{args.pred_hhsearch_out}/pdb_hits.hhr',
                                           pkl_out=f'{args.pred_hhsearch_out}/dict_file.pkl')
        feature = hhsearch_feature.call()

        pre_INPUTS = af_back_loss.prep_inputs_idx(f'{args.hhsearch_out}/msa_total.a3m', f'{args.pdb_out}/template_rank.pdb', idx=0)
        pre_INPUTS_idx = af_back_loss.prep_inputs_idx(pseudo_fa, f'{args.pred_hhsearch_out}/msa_total.a3m', f'{args.pdb_out}/template_rank.pdb', idx=0)
        pre_INPUTS_idx_af2 = af_back_loss.prep_inputs_idx(pseudo_fa, f'{args.pred_hhsearch_out}/msa_total.a3m', f'{args.pdb_out}/template_rank.pdb', idx=0)
    
    # pred_input load
        pre_msa_inputs = {"x":INPUTS_idx_af2["ms"], "aln":INPUTS_idx_af2["aln"], "lengths":INPUTS_idx_af2["lens"]}

        pre_input_hh = {'aatype':feature['aatype'],'between_segment_residues':feature['between_segment_residues'],'domain_name':feature['domain_name'],
                    'residue_index':feature['residue_index'],'seq_length':feature['seq_length'],'sequence':feature['sequence'],
                    'deletion_matrix_int':feature['deletion_matrix_int'],
                    'msa':feature['msa'],'num_alignments':feature['num_alignments'],'msa_species_identifiers':feature['msa_species_identifiers'],
                    'template_aatype':feature['template_aatype'],'template_all_atom_masks':feature['template_all_atom_masks'],
                    'template_all_atom_positions':feature['template_all_atom_positions'],'template_domain_names':feature['template_domain_names'],
                    'template_sequence':feature['template_sequence'],'template_sum_probs':feature['template_sum_probs']}

        pre_inputs = {"x":pre_INPUTS_idx["ms"], "lengths":pre_INPUTS_idx["lens"],'aln':pre_INPUTS_idx["aln"],
                  "x_ref":pre_INPUTS["ms"][:1],'x_ref_len':pre_INPUTS["lens"][:1], "key":key.get(10)}

    # pred_input AF2

    # AF2 feature updata
        model_runner, model_params = af_back_loss.get_model_runner(pre_INPUTS_idx["N"], dropout=args.dropout,backprop_recycles=args.backprop_recycles)

        feat_inputs = model_runner.process_features(input_hh, random_seed=args.seed)

    # AF2 backword
        loss_fn, grad_fn = af_back_loss.get_grad_fn_nomsa(model_runner, x_ref_len=pre_INPUTS["lens"][0],
                                 confidence=confidence, unsupervised=unsupervised, supervised=supervised,
                                 batch=pre_INPUTS_idx["batch"])
        grad_fn = jax.jit(grad_fn)

        LOSSES = []
        BEST_PLDDT = 0
        losses = []
        loss = 0

    # optimizer
        init_fun, update_fun, get_params = adam(step_size=args.lr)

        state = loaded_variables['param_smurf']
        model_params = loaded_variables['param_af2']

        (loss, outs), grad = grad_fn(get_params(state), pre_inputs['key'][0], feat_inputs, model_params, msa_inputs)
        plddt = outs["plddt"].mean()

        losses[-1].append([loss,plddt])
        losses[-1][-1].append(outs["losses"]["rmsd"])
        losses[-1][-1].append(outs["losses"]["cce"])

    # save results
        if plddt > BEST_PLDDT:
            BEST_PLDDT = plddt

        with open(f"{args.pred_pdb_output}/pred_pseudo_loss.txt", 'w') as file1:
            print(f'pred_pseudo_model\tcce:{cce_loss_pred}\tRMSD:{outs["losses"]["rmsd"]}\tplddt:{plddt}',file=file1)

        pickle.dump({"losses":losses}, open(f"{args.pred_pdb_output}/pred_pseudo_loss.pkl","wb"))

        save_pdb(outs,f"{args.test_pdb_output}/pseudo_model.pdb")

#################################################

if __name__ =='__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.hhsearch_out:
        Path(args.hhsearch_out).mkdir(parents=True, exist_ok=True)
    if args.pdb_out:
        Path(args.pdb_out).mkdir(parents=True, exist_ok=True)
    if args.test_seq:
        Path(args.test_seq).mkdir(parents=True, exist_ok=True)
    if args.test_hhsearch_out:
        Path(args.test_hhsearch_out).mkdir(parents=True, exist_ok=True)
    if args.test_pdb_output:
        Path(args.test_pdb_output).mkdir(parents=True, exist_ok=True)
    if args.pred_hhsearch_out:
        Path(args.pred_hhsearch_out).mkdir(parents=True, exist_ok=True)
    if args.pred_pdb_output:
        Path(args.pred_pdb_output).mkdir(parents=True, exist_ok=True)

    mode = 'train' # train/evalu

    if mode == 'train':
        main(args, mode=mode)
    else:
        print('test ==============================================================')
        main(args, mode=mode)

