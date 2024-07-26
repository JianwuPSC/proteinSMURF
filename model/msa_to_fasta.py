######### one hot to number

#label = [one_label.tolist().index(1) for one_label in msa] # 找到下标是1的位置
#label = [one_label.tolist().index(max(one_label)) for one_label in msa]

######## number to sequence

def one_emb(msa):
    emb=[]
    for one_label in msa:
        if 1. in one_label.tolist():
            emb.append(one_label.tolist().index(1.0))
        else:
            emb.append(20)
    return emb

########

def one_emb_v(msa):
    emb=[]
    for one_label in msa:
        if max(one_label) in one_label.tolist() and (max(one_label) != min(one_label)):
            emb.append(one_label.tolist().index(max(one_label)))
        else:
            emb.append(20)
    return emb


alphabet="ARNDCQEGHILKMFPSTWYV-"
a2n = {a:n for n,a in enumerate(alphabet)}
list_of_key = list(a2n.keys())
list_of_value = list(a2n.values())  

####  onehot to sequence
def get_fasta(x):
    seq=[]
    for aa in list(x):
        seq.append(list_of_key[(list_of_value.index(aa))])
    sequence = ''.join(seq)
    return sequence


def msa_to_fasta(feature,out1):
    
    label = [one_label.tolist().index(max(one_label)) for one_label in feature]

    with open(out1, 'w') as file1:
        seq=get_fasta(feature)
        print(f'>pred_pseudo_seq\n{seq}', file=file1)
    file1.close()

    return out1

