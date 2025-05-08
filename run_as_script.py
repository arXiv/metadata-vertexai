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
'''.strip().splitlines()
skip_ids = set(np.array(skip_ids))





if __name__ == "__main__":

    test_ids_df = pd.read_csv("gs://institutional-extract-scratch/reference/arx_ids/2311_ids.csv")
    #test_ids_df = pd.read_csv("gs://institutional-extract-scratch/reference/arx_ids/2023_all_ids.csv")
    
    #test_ids_df.head()
    ids_2311_all = test_ids_df["arx_id"].unique()
    

    tt = TicToc()

    input_ids = set(ids_2311_all)
    time_code = "2025-05-08"
    save_name = "2311_db_json"
    sample_size = "all"
    batch_size = 20      # articles
    mp_epoch_size = 20480 #batches
    parallel_workers = 6 #28 #8
    thread_workers = 10

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
        known_ids.extend(x[0] for x in obj)
        known_res_set.update(obj)
        
    known_ids = set(known_ids)
    input_ids = input_ids - known_ids
    known_res = list(known_res_set)
    print(f"Found checkpoints for {len(known_ids)} nodes.")

    if True:
        arxid_itr = itr.groupby(sorted(known_res, key=lambda x: x[0]), key=lambda x: x[0])
        no_result_idx = [arx_id for arx_id, grp in arxid_itr if all(x[1] in ('null', 'error') for x in grp)]
        rerun_ids = set(no_result_idx)
        print(f"Found {len(rerun_ids)} error nodes nodes to reprocess.")
        input_ids = input_ids.union(rerun_ids)
    
    if sample_size != "all":
        input_ids = input_ids[:sample_size]
        
    input_ids = input_ids - skip_ids

    #import concurrent.futures
    #import phase_one


    os.environ["TOKENIZERS_PARALLELISM"] = "false" 
    
    batches = [list(input_ids)]
    if len(input_ids) > batch_size:
        batches = np.array_split(list(input_ids), math.ceil(len(input_ids)/batch_size))
    print(f"Start: {len(input_ids)} in {len(batches)} batches")
    mp_batches = [batches]
    if len(batches) > mp_epoch_size:
        mp_batches = np.array_split(batches, math.ceil(len(batches)/mp_epoch_size))
    print(f"Start: {len(batches)} batches in {len(mp_batches)} epochs")
    tt.tic()
    print(f"Start: {len(input_ids)} in {len(batches)} batches")
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