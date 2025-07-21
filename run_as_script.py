import tarfile
import zipfile
import io
import os
import time
import math
import pickle
import itertools as itr
import functools as ft
import regex as re
import chardet
from tqdm.auto import tqdm
from pytictoc import TicToc

import pandas as pd
import numpy as np
import gc

import gcsfs
fs = gcsfs.GCSFileSystem()

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Phase 2
import json
import vertexai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage

from langchain.docstore.document import Document
from langchain_google_vertexai import VertexAI
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

PROJECT_ID = "arxiv-development"
vertexai.init(project=PROJECT_ID, location="us-central1")

from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexGroupNode, LatexMacroNode
from pylatexenc.latex2text import LatexNodes2Text

os.chdir("/home/jupyter/metadata-vertexai/")  # this needs to be the folder where notebook lives
import importlib
import phase_one_json as phase_one


def worker(arx_id_list):
    res = phase_one.process_tex_files(arx_id_list, max_workers=thread_workers)
    return res

def run_phase_one_in_parallel(arx_id_batches, checkpoint_fp=None):
    res_list = []
    total_len = sum(len(x) for x in arx_id_batches)
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [executor.submit(worker, arx_id_list) for arx_id_list in arx_id_batches]
        for future in tqdm(as_completed(futures), total=len(futures), ncols=100, desc='MP batch progress'):
            res = future.result()
            res_list.extend(res)
            if checkpoint_fp is not None:
                pickle.dump(res, checkpoint_fp)
#        res = [future.result() for future in concurrent.futures.as_completed(futures)]
#        for batch in res:
#            res_list.extend(batch)
    return res_list

def format_results(arxid_inst_ror_list):
    '''Out of sync with current format and not used'''
    res_list = []
    for key, group in tqdm(itr.groupby(arxid_inst_ror_list, key=lambda x: x[0])):
        ror_inst = []
        for x in group:
            clean_name = x[1].split('.', 1)[-1].strip()
            ror = 'null'
            if len(x) == 3:
                ror = x[2].strip()
            inst = {
                'name':clean_name,
                'ror_id':ror
            }
            ror_inst.append(inst)
        arx_rec = {
            "arxiv_id": key.strip(),
            "institutions_with_ror": ror_inst,
        }
        res_list.append(arx_rec)
    return res_list


skip_ids = '''
2301.08641v2
2304.09870v2
2309.01118v2
2310.20374v3
2303.11590v3
2308.04512v1
2312.05433v2
2304.14219v4
2307.05569v1
2312.07121v1
2312.14567v1
2303.01063v2
2308.04512v2
2309.08117v3
2302.07019v1
2305.04720v2
2306.03953v1
2310.13041v1
2310.19023v1
2312.05433v1
2410.12880v2
2410.12880v3
2409.14926v1
2402.00526v1
2408.12951v2
2402.19257v1
2409.14926v2
2409.07900v1
2408.16143v1
2404.04248v3
2412.13923v2
2409.14926v3
2407.18006v1
2410.15024v1
2407.12751v1
2407.18006v1
2403.09641v1
2401.13774v1
2404.16381v1
2405.20642v2
2407.18006v1
2410.19352v1
2410.12880v3
2405.16216v3
2403.00607v1
2407.01526v1
2411.08928v1
2403.08704v1
2402.10194v1
2402.17628v1
2409.19395v1
2412.03773v1
2410.08633v2
2401.15208v1
2404.10899v1
2411.08535v1
2406.01307v1
2406.17651v4
2407.18006v1
2409.09795v1
2409.10836v2
2412.01105v1
2401.13665v2
2403.00607v1
2412.03773v1
2404.03056v1
2403.12280v1
2408.10832v1
2411.13109v2
2406.17651v4
2403.17965v1
2406.11786v1
2412.20620v1
2407.12751v1
2410.20476v2
2411.19003v1
2409.09847v1
2406.17651v5
2405.17445v1
2409.10836v1
2401.14299v1
2402.10194v1
2410.12880v3
2404.16381v1
2406.02600v1
2402.12684v1
2410.17615v1
2402.06041v1
2406.00643v1
2411.01655v1
2412.05942v1
2411.19791v1
2412.07837v1
2402.13944v2
2409.01983v1
'''.strip().splitlines()
skip_ids = set(skip_ids) #set([]) #





if __name__ == "__main__":

    #test_ids_df = pd.read_csv("gs://institutional-extract-scratch/reference/arx_ids/2311_ids.csv")
    #time_code = "2025-05-09v2"
    #save_name = "2311_db_json"
    
    test_ids_df = pd.read_csv("gs://institutional-extract-scratch/reference/arx_ids/2024_all_ids.csv")
    time_code = "2025-05-12"
    save_name = "2024_db_json"
    
    sample_size = "all"
    batch_size = 20      # articles
    mp_epoch_size = 2048 #batches
    parallel_workers = 28 #8
    thread_workers = 10
    
    
    
    #test_ids_df.head()
    test_ids_df['arx_id'].apply(lambda x: x.split('v')[0]).nunique()
    split_df = pd.DataFrame.from_records(test_ids_df['arx_id'].str.split('v'), columns=['paper_id', 'version'])
    dedup_df = split_df.groupby('paper_id')['version'].max().reset_index()
    arx_ids = (dedup_df['paper_id']+'v'+dedup_df['version']).tolist()
    
    tt = TicToc()

    input_ids = set(str(x) for x in arx_ids)
    del test_ids_df
    del split_df
    del dedup_df
    del arx_ids

    try:
        objects = []
        with open(f"checkpoints/{save_name}_{sample_size}_{time_code}.pkl", 'rb') as cp_fp:
            while True:
                try:
                    obj = pickle.load(cp_fp)
                    objects.append(obj)
                except EOFError:
                    break
    except FileNotFoundError as e:
        pass
    
    known_ids = []
    known_res_set = set()
    for obj in objects:
        known_ids.extend(str(x[0]) for x in obj)
        known_res_set.update(obj)
        
    known_ids = set(known_ids)
    input_ids = input_ids - known_ids
    known_res = list(known_res_set)
    print(f"Found checkpoints for {len(known_ids)} nodes.")

    if True:
        arxid_itr = itr.groupby(sorted(known_res, key=lambda x: x[0]), key=lambda x: x[0])
        no_result_idx = [str(arx_id) for arx_id, grp in arxid_itr if all(x[1] in ('null', 'error') for x in grp)]
        rerun_ids = set(no_result_idx)
        print(f"Found {len(rerun_ids)} error nodes nodes to reprocess.")
        input_ids = input_ids.union(rerun_ids)
    
    if sample_size != "all":
        input_ids = input_ids[:sample_size]
        
    input_ids = input_ids - skip_ids

    #import concurrent.futures
    #import phase_one


    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    
    mp_batch_ids = [list(input_ids)]
    if len(input_ids)/batch_size > mp_epoch_size:
        #Split into epochs
        mp_batch_ids = np.array_split(list(input_ids), math.ceil(len(input_ids)/batch_size/mp_epoch_size))
    mp_batches = [x for x in mp_batch_ids]
    if len(input_ids) > batch_size:
        mp_batches = [
            np.array_split(x, math.ceil(len(x)/batch_size)) 
            if len(x) > batch_size
            else x 
            for x in mp_batch_ids
        ]
    num_batches = sum(len(x) for x in mp_batches)

    print(f"Start: {len(input_ids)} articles in {num_batches} batches")
    print(f"Start: {num_batches} batches in {len(mp_batches)} epochs")
    tt.tic()
    
    with open(f"checkpoints/{save_name}_{sample_size}_{time_code}.pkl", 'ab') as cp_fp:
        for mp_batch in tqdm(mp_batches, desc="MP cycles", ncols=100):
            res = run_phase_one_in_parallel(mp_batch, cp_fp)
            known_res.extend(res)
            del res
            gc.collect()
    tt.toc()
    res_df = pd.DataFrame.from_records(known_res, columns=['arx_id', 'name', 'location', 'ror']).drop_duplicates()
    res_df.to_csv(f"gs://institutional-extract-scratch/output/{save_name}_{sample_size}_{time_code}.csv.zip", index=False)
    # manually remove pickle file
    #res_list = format_results(res)
    tt.toc()