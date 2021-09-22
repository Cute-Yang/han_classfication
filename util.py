import numpy as np
from configparser import ConfigParser
import os
import json

def load_vocabulary(vocab_path,vocab_size:int)->dict:
    vocab_dict={}
    with open(vocab_path,"r",encoding="utf-8") as f:
        for index,line in enumerate(f):
            if index>=vocab_size:
                break
            line_split=line.strip().split(" ")
            word,_=line_split
            vocab_dict[word]=index
    return vocab_dict


def load_embedding(embed_path,word_dict,embed_size:int=200,vocab_size=20000)->np.ndarray:
    if len(word_dict)<vocab_size:
        raise ValueError("the size of word dict: {} is less than vocab size:{}".format(len(word_dict),vocab_size))
    print("start to load pretrain embedding vecotor")
    weight=np.random.random(size=(vocab_size,embed_size))
    with open(embed_path,"r",encoding="utf-8") as f:
        embed_info=f.readline().strip().split(" ")
        words_size,vector_size=int(embed_info[0]),int(embed_info[1])
        if vector_size!=embed_size:
            raise ValueError("you specify an invalid embed_size: {} which not match with embed_file's vector size: {}".format(embed_size,vector_size))
        for line in f:
            line_split=line.strip().split(" ")
            word,vector=line_split[0],line_split[1:]
            vector=[float(v) for v in vector]
            if word in word_dict:
                index=word_dict[word]
                weight[index,]=vector
    weight_cache="cache/embed_cache.npy"
    np.save(weight_cache,weight)
        
    print("finished to load pretrain embedding from {}".format(embed_path))



def parse_config(config_path:str,section:str,filed_name=None,encoding="utf-8"):
    cnf=ConfigParser()
    if not os.path.exists(config_path):
        raise
    
    cnf.read(config_path,encoding=encoding)
    if filed_name is None:
        config_dict=dict(cnf[section])
        return config_dict
    else:
        value=cnf.get(section,filed_name)
        return value


def read_label_json(label_json_path:str)->tuple:
    with open(label_json_path,"r",encoding="utf-8") as f:
        label_2_index=json.load(f)
    index_2_label={}
    for key,value in label_2_index.items():
        index_2_label[value]=key
    
    return label_2_index,index_2_label