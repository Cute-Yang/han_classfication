import pandas as pd
import os
import sys
from typing import Sequence
from collections import deque
import logging
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
import re

logger=logging.getLogger("parse_data")
logger.setLevel(logging.INFO)
hand=logging.StreamHandler(stream=sys.stdout)
hand.setLevel(logging.INFO)
hand.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(name)s %(message)s"))
logger.addHandler(hand)


root_data_dir="mark_data"
data_dir_depth=2

def get_files_path(root_dir:str,result:deque)->Sequence:
    files_name=os.listdir(root_dir)
    for file_name in files_name:
        file_path=os.path.join(root_dir,file_name)
        if os.path.isfile(file_path):
            result.append(file_path)
        elif os.path.isdir(file_path):
            get_files_path(file_path,result)
        else:
            print("failed to walk with {}".format(file_path))
            continue

def test_file_path():
    files_path=deque()
    get_files_path("mark_data/5月",files_path)


def get_file_prefix_suffix(file_path:str)->tuple:
    _,file_name=os.path.split(file_path)
    file_name_split=file_name.rsplit(".",1)
    if len(file_name_split)==1:
        raise ValueError("we can not get the suffix of {}".format(file_path))
    prefix,suffix=file_name_split
    return (prefix,suffix)


def convert_2_dict(pair_list:Sequence,sep:str)->dict:
    result_dict={}
    for pair in pair_list:
        pair_split=pair.split(sep,1)
        if len(pair_split)==2:
            key,value=pair_split
            result_dict[key]=value
        else:
            print("can not parse bad content: {}".format(pair))
            continue
    return result_dict

def strip_char(x:str)->str:
    try:
        x=x.replace("\r\n","").replace("\r","").replace("\n","").replace("\t","")
    except Exception as e:
        x=str(x)
    finally:
        return x

def parse_and_save_file(file_path:str,dst_file_writer:str,content_key="样本内容",label_key="打标结果",report_key="证据文本")->None:
    if "227审核" in files_path:
        return
    logger.info("start to pase file {}".format(file_path))
    _,suffix=get_file_prefix_suffix(file_path)
    
    if suffix=="csv":
        try:
            file_df=pd.read_csv(file_path,error_bad_lines=False,warn_bad_lines=False)
        except:
            try:
                file_df=pd.read_csv(file_path,error_bad_lines=False,warn_bad_lines=False,encoding="gbk")
            except:
                logger.info("can not decode file {}".format(file_path))
                return
    elif suffix=="xlsx" or suffix=="xlsm":
        try:
            file_df=pd.read_excel(file_path,engine="openpyxl",sheet_name="数据源")
        except Exception as e:
            logger.warning("can not parse file {}".format(file_path))
            logger.error(e)
            return
    elif suffix=="xls":
        file_df=pd.read_excel(file_path,engine="xlrd") # for old verison file
    
    else:
        logger.info("can not read {}".format(file_path))
        return
    #begin parse file
    
    def _parse_sample_func(sample:str)->str:
        if "<br/>" in sample:
            sample_split=sample.split("<br/>")
            #will get key:value pair
            sample_dict=convert_2_dict(sample_split,sep=":")
            return sample_dict.get(report_key) or ""

        else:
            return sample

    try:
        file_df["report_text"]=file_df[content_key].apply(_parse_sample_func)
    except:
        logger.info(file_df.columns.tolist())
        logger.info("we can not find {} column in {}".format(content_key,file_path))
        return
    keep_columns=["report_text",label_key]
    samples_df=file_df[keep_columns]
    samples_df=samples_df

    #duplicated,keep the frist commit 
    samples_df=samples_df.drop_duplicates(keep="first")
    samples_df=samples_df.applymap(strip_char)
    samples_df=samples_df.applymap(lambda x:re.sub(ILLEGAL_CHARACTERS_RE,"",x))

    rows,_=samples_df.shape
    for row in range(rows):
        row_data=samples_df.iloc[row,]
        write_row="\001".join(row_data)
        dst_file_writer.write(write_row)
        dst_file_writer.write("\n")


if __name__=="__main__":
    month_range=list(range(8,9))
    root_dirs=["mark_data/{}月".format(i) for i in month_range]
    dst_file_paths=["data/{}月.txt".format(i) for i in month_range]
    dst_file_path=dst_file_paths[0]
    dir_name,_=os.path.split(dst_file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for i in range(len(month_range)):
        root_dir=root_dirs[i]
        dst_file_path=dst_file_paths[i]
        dst_file_writer=open(dst_file_path,"w",encoding="utf-8")
        files_path=deque()
        get_files_path(root_dir,files_path)
        for file_path in files_path:
            parse_and_save_file(file_path,dst_file_writer)