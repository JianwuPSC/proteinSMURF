import os
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from absl import logging

import sys
sys.path.append('/home/wuj/data/protein_design/SMURF_protein')
from model.feature import parsers_af2

from model.tools import hhblits
from model.tools import hhsearch
from model.tools import hmmsearch
from model.tools import jackhmmer
import numpy as np
import subprocess


#input_sequence_path = f'{args.input_seq}'
#uniref90_index = f'/data/biosoft_root/alphafold_database/uniref90/uniref90.fasta'
#out_uniref90_sto = f'{args.hhsearch_out}/uniref90_hits.sto'
#out_uniref90_a3m = f'{args.hhsearch_out}/uniref90_hits.a3m'
#mgnify_index = f'/data/biosoft_root/alphafold_database/mgnify/mgy_clusters_2022_05.fa'
#out_mgnify_sto = f'{args.hhsearch_out}/mgnify_hits.sto'
#bfd_index = f'/data/biosoft_root/alphafold_database/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt'
#UniRef30_index = f'/data/biosoft_root/alphafold_database/uniref30/UniRef30_2021_03'
#out_bfd_a3m = f'{args.hhsearch_out}/bfd_uniref_hits.a3m'
#out_a3m = f'{args.hhsearch_out}/msa_total.a3m'
#pdb70_index = f'/home/wuj/data/tools/AF2/AF2/Reduced_dbs/pdb70/pdb70'
#out_hhr = f'{args.hhsearch_out}/pdb_hits.hhr'

###########

########### Runs an MSA tool
def run_msa_tool(msa_runner, input_fasta_path: str, msa_out_path: str,
                 msa_format: str, use_precomputed_msas: bool,
                 max_sto_sequences: Optional[int] = None
                 ) -> Mapping[str, Any]:
  """Runs an MSA tool, checking if output already exists first."""
  if not use_precomputed_msas or not os.path.exists(msa_out_path):
    if msa_format == 'sto' and max_sto_sequences is not None:
      result = msa_runner.query(input_fasta_path, max_sto_sequences)[0]  # pytype: disable=wrong-arg-count
    else:
      result = msa_runner.query(input_fasta_path)[0]
    with open(msa_out_path, 'w') as f:
      f.write(result[msa_format])
  else:
    logging.warning('Reading MSA from file %s', msa_out_path)
    if msa_format == 'sto' and max_sto_sequences is not None:
      precomputed_msa = parsers_af2.truncate_stockholm_msa(
          msa_out_path, max_sto_sequences)
      result = {'sto': precomputed_msa}
    else:
      with open(msa_out_path, 'r') as f:
        result = {msa_format: f.read()}
  return result

############ Hhsearch prog

class Hhsearch(object):
    def __init__(self,input_sequence_path,out_path,
                 uniref90_index='/data/biosoft_root/alphafold_database/uniref90/uniref90.fasta',out_uniref90_sto='uniref90_hits.sto',out_uniref90_a3m='uniref90_hits.a3m',
                 mgnify_index='/data/biosoft_root/alphafold_database/mgnify/mgy_clusters_2022_05.fa',out_mgnify_sto='mgnify_hits.sto',
                 bfd_index='/data/biosoft_root/alphafold_database/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt',
                 UniRef30_index='/data/biosoft_root/alphafold_database/uniref30/UniRef30_2021_03',pdb70_index='/home/wuj/data/tools/AF2/AF2/Reduced_dbs/pdb70/pdb70',
                 out_bfd_a3m='bfd_uniref_hits.a3m',out_bfd_sto='bfd_uniref_hits.sto',out_a3m='msa_total.a3m',out_sto_a3m='msa_total_sto.a3m',out_hhr='pdb_hits.hhr'):
        super(Hhsearch, self).__init__()

        self.input_sequence_path = input_sequence_path
        self.out_path = out_path
        self.uniref90_index = uniref90_index
        self.out_uniref90_sto = f'{out_path}/{out_uniref90_sto}'
        self.out_uniref90_a3m = f'{out_path}/{out_uniref90_a3m}'
        self.mgnify_index = mgnify_index
        self.out_mgnify_sto = f'{out_path}/{out_mgnify_sto}'
        self.UniRef30_index = UniRef30_index
        self.bfd_index = bfd_index
        self.pdb70_index = pdb70_index
        self.out_bfd_a3m = f'{out_path}/{out_bfd_a3m}'
        self.out_bfd_sto = f'{out_path}/{out_bfd_sto}'
        self.out_a3m = f'{out_path}/{out_a3m}'
        self.out_sto_a3m = f'{out_path}/{out_sto_a3m}'
        self.out_hhr = f'{out_path}/{out_hhr}'

    def call(self):

############ jackhmmer uniref90
        jackhmmer_uniref90_runner = jackhmmer.Jackhmmer(
            binary_path='jackhmmer',
            database_path=self.uniref90_index)

        jackhmmer_uniref90_result = run_msa_tool(
            msa_runner=jackhmmer_uniref90_runner,
            input_fasta_path=self.input_sequence_path,
            msa_out_path=self.out_uniref90_sto,
            msa_format='sto',
            use_precomputed_msas=False,
            max_sto_sequences=10000)

############# jackhmmer mgnify

        jackhmmer_mgnify_runner = jackhmmer.Jackhmmer(
            binary_path='jackhmmer',
            database_path=self.mgnify_index)
		
        jackhmmer_mgnify_result = run_msa_tool(
            msa_runner=jackhmmer_mgnify_runner,
            input_fasta_path=self.input_sequence_path,
            msa_out_path=self.out_mgnify_sto,
            msa_format='sto',
            use_precomputed_msas=False,
            max_sto_sequences=501)
	
############# hhblits bfd

        hhblits_bfd_uniref_runner = hhblits.HHBlits(
            binary_path='hhblits',
            databases=[self.bfd_index, self.UniRef30_index])

        hhblits_bfd_uniref_result = run_msa_tool(
            msa_runner=hhblits_bfd_uniref_runner,
            input_fasta_path=self.input_sequence_path,
            msa_out_path=self.out_bfd_a3m,
            msa_format='a3m',
            use_precomputed_msas=False)


############# sto trans to a3m

        msa_for_templates = jackhmmer_uniref90_result['sto']
        msa_for_templates = parsers_af2.deduplicate_stockholm_msa(msa_for_templates)
        msa_for_templates = parsers_af2.remove_empty_columns_from_stockholm_msa(
            msa_for_templates)
        uniref90_msa_as_a3m = parsers_af2.convert_stockholm_to_a3m(msa_for_templates)

        msa_for_mgnify = jackhmmer_mgnify_result['sto']
        msa_for_mgnify = parsers_af2.deduplicate_stockholm_msa(msa_for_mgnify)
        msa_for_mgnify = parsers_af2.remove_empty_columns_from_stockholm_msa(
            msa_for_mgnify)
        mgnify_msa_as_a3m = parsers_af2.convert_stockholm_to_a3m(msa_for_mgnify)

############# total a3m and save file

        aa = hhblits_bfd_uniref_result['a3m'] + uniref90_msa_as_a3m + mgnify_msa_as_a3m
        
        with open(self.out_a3m, 'w') as f:
            f.write(aa)

        with open(self.out_uniref90_a3m, 'w') as f:
            f.write(uniref90_msa_as_a3m)


############# a3m trans to hhr

        cmd = ['hhsearch', 
               '-i', self.out_uniref90_a3m, 
               '-o', self.out_hhr,
               '-d', self.pdb70_index]

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process

        print(f'============ result a3m {self.out_a3m} ; result uniref90-a3m {self.out_uniref90_a3m} ; result hhr {self.out_hhr}')

#sleep(0.2)
