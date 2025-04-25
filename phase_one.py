import tarfile
import zipfile
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

from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexGroupNode, LatexMacroNode
from pylatexenc.latex2text import LatexNodes2Text

from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import time
import vertexai
from vertexai.generative_models import GenerativeModel

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
    "4. If NO institutions can be found return exactly:\n"
    "   null\n"
    "5. If NO city is found for an institution, omit city from the output.\n"
    "6. DO NOT include explanations, descriptions, or any other text\n"
    "7. Convert any LaTeX characters to unicode.\n"
    "8. Format: Each academic institution must be numbered on a new line, exactly as follows:\n"
    "   1. Institution Name 1, City\n"
    "   2. Institution Name 2, City\n"
    "9. If an institution appears more than once, output it only once.\n"
    "10. ONLY include the numbered list of institutions OR 'null'.\n\n"
    "### INPUT TEXT:\n"
    "{input_text}\n\n"
)




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
            tqdm.write(f"Failed to read {tex_file_path} with detected encoding {detected_encoding}: {e}")
            return None

    # Remove LaTeX comments (lines starting with non-escaped %)
    content = re.sub(r"(?<!\\)%.*", "", source_text)
    res_list = []

    # try parsing latex:
    auth_macros = set([
        "author", "auth", "authors",
        "institute", "inst", "institution",
        "affiliation", "affil", "affiliations",
        "address",
        "cmsinstitute",
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
    Sends a request to the Gemini API to extract potential institution names.
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

##########################
# Process one
##########################
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
            res = None
            for i,src in enumerate(src_list_gen):
                res = query_gemini_api(src)
                if verbose:
                    print(f"{i}: {src}\n")
                    print(res)
                if res.startswith("null") or res.startswith("1. null"):
                    res = None
                    continue
                if "Institution Name" in res:
                    res = None
                    continue
                if "1. Institution 1" in res:
                    res = None
                    continue
                else:
                    break # from src_list
            if (res is not None) and not (res.startswith("null") or res.startswith("1. null")):
                return res
    except (FileNotFoundError, ClientError) as e:
        # Probably not a latex source paper
        pass
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
    paper_id = arx_id.split("v")[0]

    # Phase 1 - get names from text
    gemini_res = send_one_submission_to_gemini(arx_id, verbose=verbose)
    results = []
    found_institutions = False
    if gemini_res and gemini_res.strip().lower() != "null":
            for institution in gemini_res.split("\n"):
                clean_name = institution.strip()
                if clean_name:
                    results.append((paper_id, clean_name))
                    found_institutions = True
    if not found_institutions:
        results.append((folder, "null"))

    if lock and pbar:
        with lock:
            pbar.update(1)

    return results

def process_tex_files(article_list, max_files=None, max_workers=5):
    start_time = time.time()
    if max_files:
        article_list = article_list[:max_files]
    estimated_tex_files = len(article_list)

    results = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #, tqdm(total=estimated_tex_files, desc="Processing .tex files") as pbar:
        future_to_article = {
            executor.submit(get_single_file_results, arx_id, lock=None, pbar=None): arx_id
            for arx_id in article_list
        }

        for future in as_completed(future_to_article):
            article = future_to_article[future]
            try:
                article_results = future.result(timeout=5)
                results.extend(article_results)
            except Exception as e:
                time.sleep(.5)
                try:
                    article_results = get_single_file_results(article)
                    results.extend(article_results)
                except Exception as e2:
                    print(f"❌ Error processing article '{article}': {e}")
                    results.append((article, 'error'))

    total_time = time.time() - start_time
    print(f"✅ Total processing time: {total_time:.2f} seconds")

    return results
    
