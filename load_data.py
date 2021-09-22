import numpy as np
import json
from util import load_vocabulary



def load_data(data_path:str,max_sentence_len:int,max_word_len:int,vocab_dict,label_dict,shuffle=True):
    unknown_index=vocab_dict["<unk>"]
    pad_index=vocab_dict["<pad>"]
    samples_matrix=[]
    labels_matrix=[]
    with open(data_path,"r",encoding="utf-8") as f:
        for line in f:
            line_split=line.strip().split("\001")
            words,label=line_split
            label=label_dict[label]
            words=json.loads(words)
            n1=len(words)
            doc_list=[]
            for i in range(max_sentence_len):
                if i<n1:
                    sentence=words[i]
                    n2=len(sentence)
                    if n2>=max_word_len:
                        sentence=sentence[:max_word_len]
                    else:
                        offset=(max_word_len-n2)
                        for _ in range(offset):
                            sentence.append("<pad>")
                    sentence=[vocab_dict.get(word,unknown_index) for word in sentence]
                        
                else:
                    sentence=[pad_index for i in range(max_word_len)]
                doc_list.append(sentence)
            samples_matrix.append(doc_list)
            labels_matrix.append(label)
    samples_matrix=np.array(samples_matrix,dtype=np.int32)
    labels_matrix=np.array(labels_matrix,dtype=np.int32)

    n=len(samples_matrix)
    if shuffle:
        index_range=list(range(n))
        np.random.shuffle(index_range)
        samples_matrix=samples_matrix[index_range]
        labels_matrix=labels_matrix[index_range]
    
    #make ont hot encod
    num_classes=len(label_dict)
    onehot_lables_matrix=np.zeros(shape=(n,num_classes),dtype=np.float32)
    for i in range(n):
        mask_index=labels_matrix[i]
        onehot_lables_matrix[i,mask_index]=1.0
        
    return samples_matrix,onehot_lables_matrix


if __name__=="__main__":
    vocab_dict_path="vocab/vocab.txt"
    vocab_dict=load_vocabulary(vocab_dict_path,max_features=4000)
    with open("attention_data/label.json","r") as f:
        label_dict=json.load(f)
    
    samples,labels=load_data(
        data_path="attention_data/val.txt",
        max_sentence_len=6,
        max_word_len=10,
        vocab_dict=vocab_dict,
        label_dict=label_dict,
        shuffle=False
    )
    print(samples,labels)

    print(samples[0])
    
