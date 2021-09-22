from os import read
import numpy as np
from collections import Counter
from configparser import ConfigParser
import json

from numpy.lib.shape_base import split

data_path="data/5月.txt"

def get_label(data_path:str,sep="\001",convert_dict:dict={})->list:
    label_list=[]
    with open(data_path,"r",encoding="utf-8") as f:
        for line in f:
            line_split=line.strip().split(sep)
            try:
                _,mark=line_split
            except:
                print(len(line_split))
                print(line,line_split)
                raise
            if mark.startswith("||"):
                mark=mark.replace("||","")
            second_level_mark=convert_dict.get(mark) or mark
            label_list.append(second_level_mark)
    return label_list

def get_label_config(config_path:str,section="second-level-mark")->dict:
    cnf=ConfigParser()
    cnf.read(config_path)
    label_dict=dict(cnf[section])
    return label_dict

def convert_text_2_dict(text,sep:str="<br/>")->dict:
    text_split=text(sep)
    result={}
    for item in text_split:
        item_split=item.split(":",1)
        if len(item_split)!=2:
            continue
        key,value=item_split
        result[key]=value
    return result


def parse_data(files_path,config_path,dst_writer,max_num:int,min_num:int):
    label_dict=get_label_config(config_path)
    reverse_label_dict={}
    for key,value in label_dict.items():
        value_list=value.split(",")
        for v in value_list:
            reverse_label_dict[v]=key

    total_labels_list=[]
    for file_path in files_path:
        print("start to dream with {}".format(file_path))
        label_list=get_label(file_path,convert_dict=reverse_label_dict)
        total_labels_list.extend(label_list)    
    labels_conter=Counter(total_labels_list)

    label_array=np.array(total_labels_list,dtype=np.str)
    result_indicis=[]
    result_label_list=[]
    for label in label_dict:
        if label in labels_conter:
            size=labels_conter[label]
            if size<min_num:
                continue
            result_label_list.append(label)
            choose_indicis=np.where(label_array==label)[0]
            print("{} size: {}".format(label,min(size,max_num)))
            if size>max_num:
                n=len(choose_indicis)
                edge=list(range(n))
                np.random.shuffle(edge)
                choose_indicis=choose_indicis[edge[:max_num]]
            choose_indicis=choose_indicis.tolist()
            result_indicis.extend(choose_indicis)
    result_label_dict=dict(
        zip(
            result_label_list,range(len(result_label_list))
        )
    )
    with open("model/label.json","w",encoding="utf-8") as f:
        f.write(
            json.dumps(result_label_dict,ensure_ascii=False)
        )
    result_indicis.sort()
    label_array=label_array[result_indicis]
    n=len(result_indicis)
    print(result_indicis[:10],result_indicis[-10:])
    print("the train_data size is {}".format(len(result_indicis)))
    #write dir path
    start=0
    write_index=0
    print(reverse_label_dict)
    for file_path in files_path:
        with open(file_path,"r",encoding="utf-8") as f:
            write_size=result_indicis[write_index]
            for line in f:
                if start==write_size:
                    line_split=line.strip().split("\001")
                    report_text,_=line_split
                    label=label_array[write_index]
                    write_line="{}\001{}\n".format(report_text,label)
                    dst_writer.write(write_line)
                    write_index+=1
                    if write_index==n:
                        break
                    write_size=result_indicis[write_index]
                start+=1

dst_path="train_data/frist_level_data.txt"
files_path=["data/{}月.txt".format(i) for i in range(5,9)]
dst_writer=open(dst_path,"w")

print(files_path)
parse_data(files_path,"conf/label.config",dst_writer,20000,2000)
    

