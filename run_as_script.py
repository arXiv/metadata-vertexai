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
import phase_one


def worker(arx_id_list):
    res = phase_one.process_tex_files(arx_id_list, max_workers=thread_workers)
    return res

def run_phase_one_in_parallel(arx_id_batches, checkpoint_fp=None):
    res_list = []
    total_len = sum(len(x) for x in arx_id_batches)
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [executor.submit(worker, arx_id_list) for arx_id_list in arx_id_batches]
        for future in tqdm(as_completed(futures), total=len(futures)):
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


if __name__ == "__main__":

    test_ids_df = pd.read_csv("gs://institutional-extract-scratch/reference/arx_ids/2311_ids.csv")
    #test_ids_df.head()
    ids_2311_all = test_ids_df["arx_id"].unique()
    

    tt = TicToc()

    input_ids = set(ids_2311_all)
    time_code = "2025-04-28"
    save_name = "2311_db"
    sample_size = "all"
    batch_size = 20
    parallel_workers = 8
    thread_workers = 10

    try:
        objects = []
        with open(f"checkpoints/{save_name}_{sample_size}.pkl", 'rb') as cp_fp:
            while True:
                try:
                    obj = pickle.load(cp_fp)
                    objects.append(obj)
                except EOFError:
                    break
    except FileNotFoundError as e:
        pass
    
    known_ids = []
    known_res = []
    for obj in objects:
        known_ids.extend(x[0] for x in obj)
        known_res.extend(obj)
        
    known_ids = set(known_ids)
    input_ids = input_ids - known_ids
    print(f"Found checkpoints for {len(known_ids)} nodes.")

    if sample_size != "all":
        input_ids = input_ids[:sample_size]

    #import concurrent.futures
    #import phase_one


    os.environ["TOKENIZERS_PARALLELISM"] = "false" 

    batches = np.array_split(list(input_ids), len(input_ids)//batch_size)
    tt.tic()
    print(f"Start: {len(input_ids)} in {len(batches)} batches")
    with open(f"checkpoints/{save_name}_{sample_size}_{time_code}.pkl", 'ab') as cp_fp:
        res = run_phase_one_in_parallel(batches, cp_fp)
    tt.toc()
    known_res.extend(res)
    res_df = pd.DataFrame.from_records(known_res, columns=['arx_id', 'name', 'ror'])
    res_df.to_csv(f"gs://institutional-extract-scratch/output/{save_name}_{sample_size}_{time_code}.csv.zip", index=False)
    # manually remove pickle file
    #res_list = format_results(res)
    tt.toc()