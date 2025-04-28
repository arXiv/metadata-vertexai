import tarfile
import zipfile
import gzip
import io
import os
import itertools as itr
import functools as ft
import regex as re
import chardet
from tqdm.auto import tqdm

import pandas as pd
import numpy as np

import gcsfs
from google.cloud import storage
from google.cloud.exceptions import ClientError

from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexGroupNode, LatexMacroNode, LatexCharsNode
from pylatexenc.latex2text import LatexNodes2Text

from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading

import time
import vertexai
from vertexai.generative_models import GenerativeModel

# Phase 2
from langchain.docstore.document import Document
from langchain_google_vertexai import VertexAI
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA


PROJECT_ID = "arxiv-development"
PRD_PROJECT = 'arxiv-production'
PRD_BUCKET_LOC = 'arxiv-production-data' 

vertexai.init(project=PROJECT_ID, location="us-central1")
model = GenerativeModel("gemini-1.5-flash-002")

OLD_PROMPT_TEMPLATE = (
    "Extract the institutions that the authors of the following LaTeX document are affiliated with.\n\n"
    "### STRICT OUTPUT REQUIREMENTS:\n"
    "1. Extract ONLY the institution names associated with the authors, with no additional text.\n"
    "2. Ignore research collaborations, projects, or experiments.\n"
    "3. Format: Each academic institution must be numbered on a new line, exactly as follows:\n"
    "   1. Institution Name 1\n"
    "   2. Institution Name 2\n"
    "4. If an institution appears more than once, output it only once.\n"
    "5. If NO institutions can be found, return exactly:\n"
    "   null\n"
    "6. DO NOT include explanations, descriptions, or any other text—ONLY the numbered list or 'null'.\n\n"
    "### INPUT TEXT:\n"
    "{input_text}\n\n"
)

PROMPT_TEMPLATE = (
    "You are an expert in recognizing organization names. "
    "Identify the authors' institutions in the following INPUT TEXT below.\n\n"
    "### STRICT OUTPUT REQUIREMENTS:\n"
    "1. Extract ONLY the **top-level** institution names and the city associated with the authors.\n"
    "2. Ignore any sub-units such as departments, schools, laboratories, or colleges (e.g., 'School of ECE' or 'Department of Physics').\n"
    "3. Ignore research collaborations, projects, or experiments.\n"
    "4. Ignore email addresses in the INPUT TEXT.\n"
    "5. If NO institutions can be found return exactly:\n"
    "   null\n"
    "6. If NO city is found for an institution, omit city from the output.\n"
    "7. DO NOT include explanations, descriptions, or any other text\n"
    "8. Convert any LaTeX characters to unicode.\n"
    "9. Format: Each academic institution must be numbered on a new line, exactly as follows:\n"
    "   1. Institution Name 1, City\n"
    "   2. Institution Name 2, City\n"
    "10. If an institution appears more than once, output it only once.\n"
    "11. ONLY include the numbered list of institutions OR 'null'.\n\n"
    "### INPUT TEXT:\n"
    "{input_text}\n\n"
)

VERIFY_TEMPLATE = """
Match the institution names in the LIST_OF_NAMES with the contents of the SOURCE_TEXT.
Answer "True" if ALL the institutions in the LIST_OF_NAMES are present in the SOURCE_TEXT, otherwise answer "False"\n
Only respond with "True" or "False".
### LIST_OF_NAMES:\n
{inst_list}\n\n

### SOURCE_TEXT:\n
{source_text}\n\n
""".strip()


def find_doc_class(wrapped_file, name_match=False, sub_match=False):
    '''Search for document class related lines in a file  and return a code to represent the type'''
    doc_class_pat = re.compile(r"^\s*\\document(?:style|class)")
    sub_doc_class = re.compile(r"^\s*\\document(?:style|class).*(?:\{standalone\}|\{subfiles\})")

    for line in wrapped_file:
        if doc_class_pat.search(line):
            if name_match:
                # we can miss if there are two or more lines with documentclass
                # and the first one is not the one that has standalone/subfile
                if sub_doc_class.search(line):
                    return -99999
                return 1 #main_files[tf] = 1
            if sub_match:
                if sub_doc_class.search(line):
                    return -99999
                return 0.5
    return -0.5 if sub_match else 0 #main_files[tf] = 0

def find_main_tex_source_in_tar(tar_path, encoding='utf-8', all_found=False, with_weights=False):
    '''Identify the main Tex file in a tarfile.

    Args:
        tar_path: A gzipped tar archive of a directory containing tex source and support files.
    '''
    main_tex_names = set(["paper", "main", "ms.", "article", "manuscript", "neurips"])
    sub_tex_names = set(["appendix", "supplementary", "template"])

    client = storage.Client(project=PRD_PROJECT)
    bucket = client.bucket(PRD_BUCKET_LOC)
    blob = bucket.blob(tar_path)
    tar_bytes = blob.download_as_bytes()
    tex_files = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r') as in_tar:
        tex_files = [f for f in in_tar.getnames() if f.endswith('.tex')]

        # got one file
        if len(tex_files) == 1:
            if with_weights:
                return [(tex_files[0], 1),]
            else:
                return [tex_files[0],]

        main_files = {}
        for tf in tex_files:
            depth = len(tf.split('/')) - 1
            has_main_name = any(kw in tf for kw in main_tex_names)
            has_sub_name = any(kw in tf for kw in sub_tex_names)
            fp = in_tar.extractfile(tf)
            wrapped_file = io.TextIOWrapper(fp, newline=None, encoding=encoding) #universal newlines
            # does it have a doc class?
            # get the type
            main_files[tf] = find_doc_class(
                wrapped_file,
                name_match=has_main_name,
                sub_match=has_sub_name,
                ) - depth
            wrapped_file.close()

        # return all if asked
        if all_found and with_weights:
            return sorted(main_files.items(), key=lambda x: x[1], reverse=True)
        if all_found:
            return sorted(main_files, key=main_files.get, reverse=True)

        # got one file with doc class
        if len(main_files) == 1:
            return(main_files.keys()[0])

        # account for multi-file submissions
        return(max(main_files, key=main_files.get))

def pre_format(text):
    '''Apply some substititions to make LaTeX easier to parse'''
    source_text = (
        text
        .replace('\\}\\', '\\} \\')  # Due to escape rules \\ is equivalent to \
        .replace(')}', ') }')
        .replace(')$', ') $')
        #.replace(r'\left [', r'\left[ ')
        #.replace(r'\left (', r'\left( ')
        #.replace(r'\left \{', r'\left\{ ')
    )
    return source_text

def source_from_tar(tar_path, tex_main, encoding='utf-8'):
    client = storage.Client(project=PRD_PROJECT)
    bucket = client.bucket(PRD_BUCKET_LOC)
    blob = bucket.blob(tar_path)
    tar_bytes = blob.download_as_bytes()
    tex_files = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r') as in_tar:
        fp = in_tar.extractfile(tex_main)
        wrapped_file = io.TextIOWrapper(fp, newline=None, encoding=encoding) #universal newlines
        source_text = pre_format(wrapped_file.read())
        return source_text

def extract_texsuperscript(latex_node_list, res=None):
    '''for each superscript, get the contents of the next LatexCharsNode'''
    bailout_macros = set(['abstract', 'subsection'])
    if res is None:
        res = []
    for i, node in enumerate(latex_node_list):
        sublist = []
        #print(type(node))
        try:
            if node.macroname=='textsuperscript':
                run_started = False
                text_list = []
                for nnode in latex_node_list[i:]:
                    #print(type(nnode))
                    if isinstance(nnode, LatexCharsNode):
                        text_list.append(nnode.latex_verbatim())
                        run_started = True
                    elif isinstance(nnode, LatexMacroNode):
                        if nnode.macroname == '&':
                            text_list.append('&')
                        elif run_started:
                            break
                    elif not isinstance(nnode, LatexCharsNode):
                        if run_started:
                            break
                if text_list:
                    res.append(" ".join(text_list))
        except AttributeError:
            pass
        if isinstance(node, LatexMacroNode):
            try: 
                if node.macroname in bailout_macros:
                    break
                sublist = node.nodeargd.argnlist
                #print(sublist)
            except AttributeError:
                pass
        if isinstance(node, (LatexGroupNode, LatexEnvironmentNode)):
            try:
                sublist = node.nodelist
                #print(sublist)
            except AttributeError:
                pass
        if isinstance(node, LatexEnvironmentNode) and node.environmentname=='document':
            break
        if sublist:
            extract_texsuperscript(sublist, res)
    return res
    
def extract_pre_abstract_content(tar_path, tex_main):
    """
    Parses a .tex file:
    - Removes LaTeX comments
    - Extracts institution names (via recursive regex)
    - Extracts text before the abstract
    """
    client = storage.Client(project=PRD_PROJECT)
    bucket = client.bucket(PRD_BUCKET_LOC)
    blob = bucket.blob(tar_path)
    tar_bytes = blob.download_as_bytes()
    if tar_path.endswith(".tar.gz"):
        try:
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r') as in_tar:
                fp = in_tar.extractfile(tex_main)
                wrapped_file = io.TextIOWrapper(fp, newline=None, encoding='utf-8') #universal newlines
                source_text = pre_format(wrapped_file.read())
        except UnicodeDecodeError:
            try:
                with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r') as in_tar:
                    fp = in_tar.extractfile(tex_main)
                    raw_data = in_tar.extractfile(tex_main).peek(10000)
                    result = chardet.detect(raw_data)
                    detected_encoding = result["encoding"]
                    wrapped_file = io.TextIOWrapper(
                        fp, 
                        newline=None, 
                        encoding=detected_encoding, 
                        errors="replace"
                    ) #universal newlines
                    source_text = wrapped_file.read()
            except Exception as e:
                print(
                    f"Failed to read {tar_path}-{tex_main} with"
                    " detected encoding {detected_encoding}: {e}"
                )
                return None
    else:
        try:
            with gzip.open(io.BytesIO(tar_bytes), 'rt', encoding='utf-8') as in_gz:
                source_text = in_gz.read()
        except UnicodeDecodeError:
            try:
                with gzip.open(io.BytesIO(tar_bytes), 'rb') as in_gz:
                    raw_data = in_gz.peek(10000)
                    result = chardet.detect(raw_data)
                    detected_encoding = result["encoding"]
                with gzip.open(
                    io.BytesIO(tar_bytes),
                    'rt', 
                    encoding=detected_encoding
                ) as in_gz:
                    source_text = in_gz.read()
            except Exception as e:
                print(
                    f"Failed to read {tar_path}-{tex_main} with"
                    " detected encoding {detected_encoding}: {e}"
                )

    # Remove LaTeX comments (lines starting with non-escaped %)
    content = re.sub(r"(?<!\\)%.*", "", source_text)
    #res_list = []

    # try parsing latex:
    auth_macros = set([
        "author", "auth", "authors",
        "institute", "inst", "institution",
        "affiliation", "affil", "affiliations",
        "address",
        "cmsinstitute",
    ])
    supstr = set([
        "\\textsuperscript",
    ])
    latex_extracted_institutions = []
    try:
        lxwkr = LatexWalker(content)
        (nodelist, pos, len_) = lxwkr.get_latex_nodes()
        focus_nodes = [
          (i,node) for i,node in enumerate(nodelist)
          if hasattr(node, "macroname") and node.macroname in auth_macros
        ]
        if focus_nodes:
            for i,node in focus_nodes:
                latex_extracted_institutions.append(node.latex_verbatim())
                try:
                    follow_node = nodelist[i+1]
                    if isinstance(follow_node, LatexGroupNode):
                        latex_extracted_institutions.append(follow_node.latex_verbatim())
                except IndexError:
                    pass
            if any(pat in lx for lx in latex_extracted_institutions for pat in supstr):
                sup_res = extract_texsuperscript(nodelist)
                latex_extracted_institutions.extend(sup_res)
        else:
            doc = [
                node for node in nodelist
                if isinstance(node, LatexEnvironmentNode) and node.environmentname=='document'
            ]
            if doc:
                focus_doc_nodes = [
                  (i,node) for i, node in enumerate(doc[0].nodelist)
                  if isinstance(node, LatexMacroNode) and node.macroname in auth_macros
                ]
                for i, node in focus_doc_nodes:
                    latex_extracted_institutions.append(node.latex_verbatim())
                    try:
                        follow_node = doc[0].nodelist[i+1]
                        if isinstance(follow_node, LatexGroupNode):
                            latex_extracted_institutions.append(follow_node.latex_verbatim())
                    except IndexError:
                        pass
                if any(pat in lx for lx in latex_extracted_institutions for pat in supstr):
                    sup_res = extract_texsuperscript(doc[0].nodelist)
                    latex_extracted_institutions.extend(sup_res)
        if latex_extracted_institutions:
            #res_list.append(latex_extracted_institutions)
            yield "\n".join(latex_extracted_institutions)
    except Exception as e:
        print(f"Overly broad except in extract_pre_abstract_content(): {e}")
        pass
    
    #  "recursive" regex:
    #   ((?>[^{}]+|\{(?1)\})*)
    # optional brackets
    #   (:?\[\d+\])?\s*
    # This matches text possibly containing normal characters or nested braces,
    # until the outermost braces are matched.
    # If your LaTeX does not have deep nesting, this mainly ensures things like $^{1}$ are correctly parsed.
    institution_patterns = [
        r"\\affiliation\s*(:?\[\d+\])?\s*\{((?>[^{}]+|\{(?1)\})*)\}",
        r"\\institute\s*(:?\[\d+\])?\s*\{((?>[^{}]+|\{(?1)\})*)\}",
        r"\\address\s*(:?\[\d+\])?\s*\{((?>[^{}]+|\{(?1)\})*)\}",
        r"\\inst\s*(:?\[\d+\])?\s*\{((?>[^{}]+|\{(?1)\})*)\}",
        r"\\affil\s*(:?\[\d+\])?\s*\{((?>[^{}]+|\{(?1)\})*)\}",
        r"\\author\s*(:?\[\d+\])?\s*{[^}]+}{([^}]+)}",
        r"\\cmsinstitute\s*(:?\[\d+\])?\s*{[^}]+}{([^}]+)}",
    ]

    extracted_institutions = []
    for pattern in institution_patterns:
        # Use regex.findall with DOTALL to allow '.' to match newlines
        matches = re.findall(pattern, content, flags=re.DOTALL)
        if matches:
            # Strip each match and add to list
            for m in matches:
                if isinstance(m, tuple):
                    extracted_institutions.append(" ".join(m_i for m_i in m))
                else:
                    extracted_institutions.extend(m.strip() for m in matches if m.strip())

    # If any institution info is extracted, return the deduplicated joined text
    if extracted_institutions:
        # You can change the join method; here we join by newline and use set to deduplicate
        #return "\n".join(set(extracted_institutions))
        #res_list.append("\n".join(set(extracted_institutions)))
        yield "\n".join(set(extracted_institutions))

    # If no institution found, try extracting the text before the abstract
    match = re.split(
        r"\\begin\s*{\s*abstract\s*}|\\s*\\section\s*{\s*Abstract\s*}",
        content,
        maxsplit=1,
        flags=re.IGNORECASE
    )
    if len(match) > 1:
        #return match[0].strip()
        #res_list.append(match[0].strip())
        yield match[0].strip()

    # If still not found, return the first 1/3 of the content as a fallback
    content_length = len(content)
    if content_length > 0:
        one_third_length = max(content_length//3, 2000)
        #return content[:one_third_length].strip()
        #res_list.append(content[:one_third_length].strip())
        yield content[:one_third_length].strip()

    # If still not found, return an empty string
    #if res_list:
    #  yield res_list
    #else:
    # yield ["",]
    return None


def extract_select_pages_from_txt(txt_path):
    """
    Given the gs bucket path to the plain text file, return the
    first 2 pages, then the second to last, and finally the last page.

    Args:
        file_contents (str): The full text content of the paper.

    Returns:
        list of page contents
    """
    client = storage.Client(project=PRD_PROJECT)
    bucket = client.bucket(PRD_BUCKET_LOC)
    blob = bucket.blob(txt_path)
    txt_bytes = blob.download_as_bytes()
    file_contents = txt_bytes.decode('utf-8')

    # Split the text by form feed (page break)
    contents = file_contents.split("\u000C")

    page_list = [ contents[0:2] ]
    if len(contents) >= 2:
        page_list.append(contents[-2])
    if len(contents) >= 1:
        page_list.append(contents[-1])

    return page_list


def query_gemini_api(input_text):
    """
    Sends a request to the Gemini API to judge quality of result.
    """
    prompt = PROMPT_TEMPLATE.format(input_text=input_text)

    start_time = time.time()
    response = model.generate_content(prompt)
    end_time = time.time()

    timecost = end_time - start_time

    if response and response.text:
        clean_response = response.text
        # print(f"Execution time: {timecost:.4f} seconds")
        return clean_response
    else:
        print("API request failed or empty response")
        return None

def verify_with_gemini_api(inst_list, source_text, template=None):
    """
    Sends a request to the Gemini API to extract potential institution names.
    """
    if template:
        VERIFY_TEMPLATE = template
    prompt = VERIFY_TEMPLATE.format(inst_list=inst_list, source_text=source_text)

    start_time = time.time()
    response = model.generate_content(prompt)
    end_time = time.time()

    timecost = end_time - start_time

    if response and response.text:
        clean_response = response.text
        # print(f"Execution time: {timecost:.4f} seconds")
        return clean_response
    else:
        print("API request failed or empty response")
        return None
    
##########################
# Process one
##########################
BAD_IN_PATTERNS = set([
    "Variable not found",
    "Variable, null",
    "Institution Name",
    "1. Institution 1",  
])
BAD_START_PATTERNS = set([
    "null",
    "1. null",  
])
def check_src_list_with_gemini(src_list_gen, verbose=False):
    for i,src in enumerate(src_list_gen):
        res = query_gemini_api(src)
        if verbose:
            print(f"{i}: {src}\n")
            print(res)
        if any((res.startswith(pat)) for pat in BAD_START_PATTERNS):
            res = None
            continue
        if any((pat in res) for pat in BAD_IN_PATTERNS):
            res = None
            continue
        else:
            break # from src_list
    return res    


def send_one_submission_to_gemini(arx_id, verbose=False):
    yymm = arx_id.split(".")[0]
    paper_id = arx_id.split("v")[0]
    tar_path = f"ftp/arxiv/papers/{yymm}/{paper_id}.tar.gz"
    if verbose:
        print(f"Processing {tar_path}")
    try:
        candidate_files = find_main_tex_source_in_tar(tar_path, all_found=True)
        for c_file in candidate_files:
            src_list_gen = extract_pre_abstract_content(tar_path, c_file)
            res = check_src_list_with_gemini(src_list_gen, verbose=verbose)
            if (res is not None) and not (res.startswith("null") or res.startswith("1. null")):
                return res
    except (FileNotFoundError, ClientError) as e:
        # Probably single file latex or not a latex source paper
        pass
    ## Is it gz?
    gz_path = f"ftp/arxiv/papers/{yymm}/{paper_id}.gz"
    if verbose:
        print(f"Processing {gz_path}")
    try:
        src_list_gen = extract_pre_abstract_content(gz_path, None)
        res = check_src_list_with_gemini(src_list_gen, verbose=verbose)
        if (res is not None) and not (res.startswith("null") or res.startswith("1. null")):
            return res
    except (ClientError):
        pass
    # check text
    txt_path = f'txt/arxiv/{yymm}/{arx_id}.txt'
    if verbose:
        print(f"Processing {txt_path}")
    src_list = extract_select_pages_from_txt(txt_path)
    res = None
    for src in src_list:
        res = query_gemini_api(src)
        if verbose:
            print(res)
        if res.startswith("null") or res.startswith("1. null"):
            res = None
        else:
            break
    if (res is None) or (res.startswith("1. null")): res = "null"

    return res

##########################
# Threaded processing for multiple files
##########################
def get_single_file_results(arx_id, lock=None, pbar=None, verbose=False):
    #paper_id = arx_id.split("v")[0]

    # Phase 1 - get names from text + Phase 2
    gemini_res = send_one_submission_to_gemini(arx_id, verbose=verbose)
    results = []
    found_institutions = False
    if gemini_res and gemini_res.strip().lower() != "null":
            for institution in gemini_res.split("\n"):
                clean_name = institution.split('.', 1)[-1].strip()
                if clean_name:
                    ror = ROR_FINDER.get_ror(clean_name)
                    results.append((arx_id, clean_name, ror))
                    found_institutions = True
    if not found_institutions:
        results.append((arx_id, "null", "null")) 
    
    if lock and pbar:
        with lock:
            pbar.update(1)

    return results

def process_tex_files(article_list, max_files=None, max_workers=5, verbose=False):
    start_time = time.time()
    if max_files:
        article_list = article_list[:max_files]
    estimated_tex_files = len(article_list)

    results = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #, tqdm(total=estimated_tex_files, desc="Processing .tex files") as pbar:
        future_to_article = {
            executor.submit(get_single_file_results, arx_id, lock=None, pbar=None): str(arx_id)
            for arx_id in article_list
        }
        successes = []
        try:
            for future in as_completed(future_to_article, timeout=120):
                article = future_to_article[future]
                try:
                    article_results = future.result(timeout=5)
                    results.extend(article_results)
                    successes.append(article)
                except Exception as e:
                    time.sleep(.5)
                    try:
                        article_results = get_single_file_results(article)
                        results.extend(article_results)
                        successes.append(article)
                    except Exception as e2:
                        if verbose: 
                            print(f"❌ Error processing article '{article}': {e}")
                        results.append((article, 'error'))
        except TimeoutError as e_time:
            failures = [
                a_id for a_id in future_to_article.values()
                if not a_id in set(successes)
            ]
            for article in failures:
                results.append((article, 'error'))
                if verbose:
                    print(f"❌ Error processing article '{failures}': {e_time}")

    total_time = time.time() - start_time
    if verbose:
        print(f"✅ Total processing time: {total_time:.2f} seconds")

    return results
    
############################
## Phase 2
#############################
def load_special_cases_ror():
    docs = []
    ror_sp_gspath = 'gs://institutional-extract-scratch/reference/special_cases.json'
    fs = gcsfs.GCSFileSystem()
    try:
        with fs.open(ror_sp_gspath, "r", encoding="utf-8") as f:
            spec_data = json.load(f)
        for entry in spec:
            ror_id = entry.get("ror_id", "")
            name_loc = entry.get("name_loc", "")
            if name and ror_id:
                content = f"{name_loc} — {ror_id}"
                docs.append(Document(page_content=content))
    except FileNotFoundError:
        docs = []
    return docs

ROR_TEMPLATE = """
You are given an input institution name and a list of ROR entries.

Institution: {question}

Context:
{context}

From the context, pick the best matching ROR ID. If none match, return "null".
DO NOT include explanations, descriptions, or any other text — ONLY the ROR ID or 'null'.
Answer:
"""

class rorFinder:
    
    def __init__(self):
        self.qa_chain = self.build_qa_chain()
        self.ror_cache = {}

    @staticmethod
    def build_qa_chain():
        # Load the index model, training it if needed.
        model_project = 'arxiv-development'
        model_bucket_loc = 'institutional-extract-scratch'
        dest_blob_name = "models/ror_index.zip"
        local_index = "ror_index"
        #os.chdir("/home/jupyter/metadata-vertexai/")


        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        client = storage.Client(project=model_project)
        bucket = client.bucket(model_bucket_loc)
        blob = bucket.blob(dest_blob_name)
        RECREATE_INDEX = False

        if RECREATE_INDEX or (not blob.exists()) :
            ror_gspath = 'gs://institutional-extract-scratch/reference/v1.63-2025-04-03-ror-data_schema_v2.json'
            fs = gcsfs.GCSFileSystem()
            with fs.open(ror_gspath, "r", encoding="utf-8") as f:
                ror_data = json.load(f)


            docs = []
            docs = load_special_cases_ror()
            for i,entry in enumerate(ror_data):
                ror_id = entry.get("id", "")
                locs = entry.get('locations')
                loc_name = ""
                try:
                    loc_name = f", {locs[0]['geonames_details']['name']}"
                except KeyError:
                    pass
                for name_info in entry.get("names", []):
                    name = name_info.get("value", "")
                    if name and ror_id:
                        # format: "name — ROR_id"
                        content = f"{name}{loc_name} — {ror_id}"
                        docs.append(Document(page_content=content))

            print(f"Prepared {len(docs)} vector entries to build FAISS index")

            # Embedding model (recommended: MiniLM)
            embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            # Build FAISS index
            vectorstore = FAISS.from_documents(docs, embedding_model)

            # Save index to local file
            vectorstore.save_local(local_index)
            print("ROR vector index built and saved successfully")

            with zipfile.ZipFile(local_index+'.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(local_index):
                    for file in files:
                        full_path = os.path.join(root, file)
                        zipf.write(full_path, os.path.relpath(full_path, local_index))

            client = storage.Client(project=model_project)
            bucket = client.bucket(model_bucket_loc)
            blob = bucket.blob(dest_blob_name)
            blob.upload_from_filename(local_index+'.zip')

            print(f'File uploaded to {dest_blob_name}')

        else:
            if not os.path.exists(local_index):
                blob.download_to_filename(local_index+'.zip')
                with zipfile.ZipFile(local_index+'.zip', 'r') as zipf:
                    zipf.extractall(local_index)

            vectorstore = FAISS.load_local(
                local_index,
                embeddings=embedding,
                allow_dangerous_deserialization=True
            )

        llm = VertexAI(
        model_name="gemini-1.5-flash-002",   # "gemini-1.5-flash-002"
        temperature=0,
        max_output_tokens=512,
        )

        prompt = PromptTemplate(input_variables=["question", "context"], template=ROR_TEMPLATE)

        #  Build a Retrieval + QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        return qa_chain

    #@ft.cache # tends to cache issues too, so we rolled out own
    def get_ror(self, inst_name):
        cache_lookup = self.ror_cache.get(inst_name)
        if cache_lookup:
            return cache_lookup
        try:
            response = self.qa_chain.invoke({"query": inst_name})
            ror_id = response["result"].strip()
            if ror_id == 'null':
                #try without city
                inst_name_sans = ','.join(inst_name.split(',')[:-1])
                response = self.qa_chain.invoke({"query": inst_name_sans})
                ror_id = response["result"].strip()
            if ror_id != 'null':
                self.ror_cache[inst_name] = ror_id

        except Exception as e:
            print(f"Error querying {inst_name}: {e}")
            ror_id = "error"
        return ror_id.strip()
    
ROR_FINDER = rorFinder()