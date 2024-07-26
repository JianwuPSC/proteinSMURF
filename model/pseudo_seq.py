import random

alphabet="ARNDCQEGHILKMFPSTWYV-"

def random_word(sentence):

    tokens = []
    for i, token in enumerate(list(sentence)):
        tokens.append(token)
    output_label = []

    for i, token in enumerate(list(tokens)):
        prob = random.random()
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = '-'

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(alphabet)

            # 10% randomly change token to current token
            else:
                tokens[i] = token

            output_label.append(token)

        else:
            tokens[i] = tokens[i]
            output_label.append('*')

    return tokens,output_label


#############################################

def generate_random(sentence,out1,out2):

    with open(sentence, "r") as f:
        input_sequence = f.read().split("\n")[1]
        input_description = f.read().split("\n")[0]

    with open(out1, 'w') as file1, open(out2, 'w') as file2:

        tokens, output_label=random_word(input_sequence)
        tokens_str=''.join(tokens)
        output_label=''.join(output_label)

        print(f'>pseudo_seq{i}\n{str(tokens)}', file=file1)
        print(f'>true_seq{i}\n{str(output_label)}', file=file2)

    file1.close()
    file2.close()
    
    return out1,out2

