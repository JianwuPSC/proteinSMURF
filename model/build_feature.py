import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union

from alphafold.common import residue_constants

import sys
sys.path.append('/home/wuj/data/protein_design/SMURF_protein')

from model.feature import parsers_af2
from model.feature import msa_identifiers_af2
from model.feature import templates_af2

import pickle
from model.tools import hhsearch

#seq_path=args.input_seq
#a3m_path=args.hhsearch_out/msa_total.a3m
#hhr_path=args.hhsearch_out/pdb_hits.hhr
#pkl_out=args.hhsearch_out/dict_file.pkl

class Hhsearch_feature(object):
    def __init__(self, seq_path,a3m_path='msa_total.a3m', hhr_path='pdb_hits.hhr', pkl_out='dict_file.pkl'):

        self.seq_path = seq_path
        self.a3m_path = a3m_path
        self.hhr_path = hhr_path
        self.pkl_out = pkl_out
    def call(self):
################ input seq_feature

        with open(self.seq_path, "r") as f:
            input_sequence = f.read().split("\n")[1]
            input_description = f.read().split("\n")[0]
        num_res=len(input_sequence)

        FeatureDict = MutableMapping[str, np.ndarray]
        def make_sequence_features(
            sequence: str, description: str, num_res: int) -> FeatureDict:
            """Constructs a feature dict of sequence features."""
            features = {}
            features['aatype'] = residue_constants.sequence_to_onehot(
                sequence=sequence,
                mapping=residue_constants.restype_order_with_x,
                map_unknown_to_x=True)
            features['between_segment_residues'] = np.zeros((num_res,), dtype=np.int32)
            features['domain_name'] = np.array([description.encode('utf-8')],
                                      dtype=np.object_)
            features['residue_index'] = np.array(range(num_res), dtype=np.int32)
            features['seq_length'] = np.array([num_res] * num_res, dtype=np.int32)
            features['sequence'] = np.array([sequence.encode('utf-8')], dtype=np.object_)
            return features

        sequence_features = make_sequence_features(
  	    sequence=input_sequence,
  	    description=input_description,
  	    num_res=num_res)

################### MSA feature

        def make_msa_features(msas: Sequence[parsers_af2.Msa]) -> FeatureDict:
            """Constructs a feature dict of MSA features."""
            if not msas:
                raise ValueError('At least one MSA must be provided.')
            int_msa = []
            deletion_matrix = []
            species_ids = []
            seen_sequences = set()
            for msa_index, msa in enumerate(msas):
                if not msa:
                    raise ValueError(f'MSA {msa_index} must contain at least one sequence.')
                for sequence_index, sequence in enumerate(msa.sequences):
                    if sequence in seen_sequences:
                        continue
                    seen_sequences.add(sequence)
                    int_msa.append(
                        [residue_constants.HHBLITS_AA_TO_ID[res] for res in sequence])
                    deletion_matrix.append(msa.deletion_matrix[sequence_index])
                    identifiers = msa_identifiers_af2.get_identifiers(
                        msa.descriptions[sequence_index])
                    species_ids.append(identifiers.species_id.encode('utf-8'))

                num_res = len(msas[0].sequences[0])
                num_alignments = len(int_msa)
                features = {}
                features['deletion_matrix_int'] = np.array(deletion_matrix, dtype=np.int32)
                features['msa'] = np.array(int_msa, dtype=np.int32)
                features['num_alignments'] = np.array(
                    [num_alignments] * num_res, dtype=np.int32)
                features['msa_species_identifiers'] = np.array(species_ids, dtype=np.object_)
                return features

        with open(self.a3m_path, 'r') as f:
            a3m_str = f.read()


######## msa feature
    
        uniref90_msa = parsers_af2.parse_a3m(a3m_str)
        msa_features = make_msa_features([uniref90_msa])

################## templete feature

        hhsearch_binary_path = 'hhsearch'
        pdb70_database_path = '/data/biosoft_root/alphafold_database/pdb70/pdb70'  # need cif;

        # searcher:
        template_searcher = hhsearch.HHSearch(binary_path=hhsearch_binary_path, databases=[pdb70_database_path])

        # 使用a3m数据类型进行:
        #pdb_templates_result = template_searcher.query(a3m_str)
        #pdb_template_hits = template_searcher.get_template_hits(output_string=pdb_templates_result, input_sequence=input_sequence)

        with open(self.hhr_path, 'r') as f:
            hhr_str = f.read()

        pdb_template_hits = template_searcher.get_template_hits(output_string=hhr_str, input_sequence=input_sequence)

        # template featurize: 
        max_template_date_ = '2022-01-01'
        MAX_TEMPLATE_HITS = 20
        kalign_binary_path_ = 'kalign'
        obsolete_pdbs_path_ = None  # 用于可自定义模板;
        template_mmcif_dir = '/data2/wuj/tools/AF2/AF2/Reduced_dbs/pdb_mmcif/mmcif_files'

        # featureizer:
        template_featurizer = templates_af2.HhsearchHitFeaturizer(
            mmcif_dir=template_mmcif_dir,
            max_template_date=max_template_date_,
            max_hits=MAX_TEMPLATE_HITS,
            kalign_binary_path=kalign_binary_path_,
            release_dates_path=None,
            obsolete_pdbs_path=obsolete_pdbs_path_)

        templates_result = template_featurizer.get_templates(query_sequence=input_sequence, hits=pdb_template_hits)
        
        # get features:
        feature_dict = {**sequence_features, **msa_features, **templates_result.features}
        
        # download feature dict
        f_save = open(self.pkl_out, 'wb')
        pickle.dump(feature_dict, f_save)
        f_save.close()

        return feature_dict
        
        print(f'===============result fearue dict {pkl_out}')

#sleep(0.2)
