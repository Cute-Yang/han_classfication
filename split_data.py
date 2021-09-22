from pickle import encode_long
import jieba
import re
from collections import Counter
import os
import random
import time
import json

data_path="train_data/first_level_data.txt"


def clean_report(report:str,split_short=True)->list:
    report=report.replace("🐮","牛")
    short_p=r"[，。？！]"
    p1=r"appmsg start>title:微信红包 des:我给你发了一个红包，赶紧去拆!"
    p2=r"<appmsg start>title:微信转账 des:收到转账￥\d{1,}\.\d{2,2}元。"
    p3=r"<appmsg start>title:微信转账 des:收到转账\d{1,}\.\d{2,2}元。"
    p5=r"我.*?现在我们可以开始聊天了|@.*?\u2005|我是.*?\u2005|appmsg|start|title|des|end|info|如需收钱，请点此升级至最新版本|如需收钱，请点击升级至最新版本|祝：恭喜发财，大吉大利"
    p6=r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"
    p7=r"\[.*?\]"
    for p in (p1,p2,p3,p5,p6,p7):
        report=re.sub(p,"",report)
        # print(report,p)
    report=report.lower()
    

    # re.sub(r'(?<=long.file.name)(\_a)?(?=\.([^\.]*)$)' , r'_suff',"long.file.name.jpg")
    report=re.sub(r"([1-9]\d*\.*\d*)(?=[\u4e00-\u9fa5])","digit",report)


    report_split=report.split("|")
    if len(report_split)==0:
        return []
    #replace 1,2.. to digit 
    
    p=r"[^a-z\u4e00-\u9fa5]"
    result=[]
    if not split_short:
        for report_item in report_split:
            item_split=report_item.split(":",1)
            if len(item_split)!=2:
                continue
            _,sentence=item_split
            if sentence=="":
                continue
            sentence=re.sub(p,"",sentence)
            if re.match(r"^[a-zA-Z]*$",sentence):
                continue
            result.append(sentence)
    else:
        for report_item in report_split:
            item_split=report_item.split(":",1)
            if len(item_split)!=2:
                continue
            _,sentence=item_split
            if sentence=="":
                continue
            sentence_split=re.split(short_p,sentence)
            for short_sentence in sentence_split:
                short_sentence=re.sub(p,"",short_sentence)
                result.append(short_sentence)
    return result
    



#load data and return train_data,validation_data,test_data
def split_data_train_val_test(data_path:str,stopwords_path:str=None,userdict_path:str=None,vocab_path:str="vocab.txt")->tuple:
    try:
        stopwords=[line.strip() for line in open(stopwords_path,"r",encoding="utf-8")]
    except:
        print("failed to load stopwords from {}".format(stopwords_path))
        stopwords=[]

    try:
        with open(userdict_path,"r",encoding="utf-8") as f:
            for line in f:
                word=line.strip()
                jieba.add_word(word)
    except:
        print("failed to load userdict word from {}".format(userdict_path))


    vocabulary=[]
    samples_words=[]
    samples_label=[]
    min_words=2
    with open(data_path,"r",encoding="utf-8") as f:
        for line in f:
            line_split=line.strip().split('\001')
            text,label=line_split
            text_list=clean_report(text,split_short=False)
            text_words=[]
            for sentence in text_list:
                sentence_words=jieba.lcut(sentence)
                sentence_words=[word for word in sentence_words if word not in stopwords]
                if len(sentence_words)<=1:
                    continue
                vocabulary.extend(sentence_words)
                text_words.append(sentence_words)
            if len(text_words)>0:
                samples_words.append(text_words)
                samples_label.append(label)
    
    #create vocabulary dict
    vocabulary_counter=Counter(vocabulary)
    labels_conter=Counter(samples_label)
    labels_unique=list(labels_conter.keys())
    labels_dict=dict(zip(labels_unique,range(len(labels_unique))))
    with open("attention_data/label.json","w",encoding="utf-8") as f:
        f.write(
            json.dumps(labels_dict,ensure_ascii=False)
        )
    n=len(vocabulary)
    min_size=5
    #filter the words freq <=5
    vocabulary_list=[]
    for v in vocabulary_counter:
        freq=vocabulary_counter[v]
        if freq<=min_size:
            continue
        vocabulary_list.append([v,freq])
    
    vocabulary_list_sorted=sorted(
        vocabulary_list,
        key=lambda x:x[1],
        reverse=True
    )
                    
    vocab_dir,_=os.path.split(vocab_path)
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)

    unknown_word="<unk>"
    pad_word="<pad>"
    unknown_index=1
    pad_index=0
    with open(vocab_path,"w",encoding="utf-8") as f:
        f.write("{} {}\n".format(pad_word,pad_index))
        f.write("{} {}\n".format(unknown_word,unknown_index))
        for item,freq in vocabulary_list_sorted:
            write_line="{} {}\n".format(item,freq)
            f.write(write_line)
    
    samples_size=len(samples_label)
    train_size=int(samples_size*0.8)
    val_size=int(samples_size*0.1)
    test_size=samples_size-(train_size+val_size)
    
    index_range=list(range(samples_size))
    random.shuffle(index_range)

    train_samples_index=index_range[:train_size]
    val_samples_index=index_range[train_size:(train_size+val_size)]
    test_samples_index=index_range[(train_size+val_size):]
    
    
    train_samples=[samples_words[item] for item in train_samples_index]
    train_labels=[samples_label[item] for item in train_samples_index]

    val_samples=[samples_words[item] for item in val_samples_index]
    val_labels=[samples_label[item] for item in val_samples_index]

    test_samples=[samples_words[item] for item in test_samples_index]
    test_labels=[samples_label[item] for item in test_samples_index]

    def _restore_samples(samples,labels,restore_path):
        size=len(samples)
        restore_dir=os.path.dirname(restore_path)
        if not os.path.exists(restore_dir):
            os.makedirs(restore_dir)
        with open(restore_path,"w") as  f:
            for i in range(size):
                sample,label=samples[i],labels[i]
                write_line="{}\001{}\n".format(
                    json.dumps(sample,ensure_ascii=False),
                    label
                )
                f.write(write_line)
    
    _restore_samples(train_samples,train_labels,"attention_data/train.txt")
    _restore_samples(val_samples,val_labels,"attention_data/val.txt")
    _restore_samples(test_samples,test_labels,"attention_data/test.txt")

    

if __name__=="__main__":
    split_data_train_val_test(
        data_path="choose_data/tiny.txt",
        stopwords_path="",
        userdict_path="",
        vocab_path="assits/vocabulary.txt"
    )
