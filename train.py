import json
import os

from han_attention_with_mask import create_HAN
from util import (
    load_vocabulary,
    load_embedding,
    parse_config,
    read_label_json
)
import numpy as np
from load_data import load_data
model_config_path="conf/model.conf"
model_config=parse_config(model_config_path,section="HAN")
print(model_config)

#reading train params
embed_size=int(model_config["embed_size"])
hidden_size=int(model_config["hidden_size"])
attention_size=int(model_config["attention_size"])
vocab_size=int(model_config["vocab_size"])
vocab_path=model_config["vocab_path"]
num_classes=int(model_config["num_classes"])
pad=model_config["pad"]
unknown=model_config["unknown"]
stopwords_path=model_config["stopwords"]
userdict_path=model_config["userdict"]
embed_file=model_config["embed_path"]
batch_size=int(model_config["batch_size"])
epochs=int(model_config["epochs"])
maxlen_sentence=int(model_config["maxlen_sentence"])
maxlen_word=int(model_config["maxlen_word"])
label_file_path=model_config["label_map_file"]
embed_path=model_config["embed_path"]

vocab_dict=load_vocabulary(vocab_path=vocab_path,vocab_size=vocab_size)
label_2_index,index_2_label=read_label_json(label_file_path)
x_train,y_train=load_data(
    data_path="attention_data/val.txt",
    max_sentence_len=maxlen_sentence,
    max_word_len=maxlen_word,
    vocab_dict=vocab_dict,
    label_dict=label_2_index,
    shuffle=True
)

x_val,y_val=load_data(
    data_path="attention_data/test.txt",
    max_sentence_len=maxlen_sentence,
    max_word_len=maxlen_word,
    vocab_dict=vocab_dict,
    label_dict=label_2_index
)


if os.path.exists("cache/embed_cache.npy"):
    print("load embeding from cache")
    pretrain_embedding=np.load("cache/embed_cache.npy")
else:
    pretrain_embedding=load_embedding(
        embed_path=embed_path,
        word_dict=vocab_dict,
        embed_size=embed_size,
        vocab_size=vocab_size
    )

han_classfier,han_classfier_prob=create_HAN(
    maxlen_word=maxlen_word,
    maxlen_sentence=maxlen_sentence,
    vocab_size=vocab_size,
    hidden_size=hidden_size,
    atten_size=attention_size,
    embed_size=embed_size,
    num_classes=num_classes,
    pretrian_embedding=pretrain_embedding
)

han_classfier.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs
)


predict_data=han_classfier_prob.predict(x_train)
score,word_prob,sentence_prob=predict_data
score=np.argmax(score,axis=1)
predict_label=[index_2_label[i] for i in score]

print(word_prob[0,1])