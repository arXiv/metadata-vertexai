import tarfile
import zipfile
import gzip
import io
import os
import gc
import itertools as itr
import functools as ft
import regex as re
import chardet
from tqdm.auto import tqdm

#import pandas as pd
#import numpy as np
import json

import gcsfs
from google.cloud import storage
from google.cloud.exceptions import ClientError
from google.resumable_media.common import InvalidResponse
from google.api_core.exceptions import GoogleAPIError, NotFound, Forbidden


from pylatexenc.latexwalker import LatexWalker, LatexEnvironmentNode, LatexGroupNode, LatexMacroNode, LatexCharsNode
from pylatexenc.latex2text import LatexNodes2Text

from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import threading

import time
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig

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
model = GenerativeModel("gemini-2.0-flash-lite")
GEN_CONFIG =  GenerationConfig(
    temperature=0.0,  # Lower = more deterministic
    top_p=0.8,        # Lower = more focused, higher = more diverse
)

#model = GenerativeModel("gemini-1.5-flash-002") # retired


### V7 JSON
PROMPT_TEMPLATE = """
TASK: Follow the directions to generate output from the SOURCE_TEXT as descibed in the OUTPUT_FORMAT directions.
Follow the directions below:
 - Find all potential organizations in the SOURCE_TEXT.
 - Expand abbreviations and acronyms of potential organization names using context for known full forms.
 - When organizations are listed together at an address, treat each organization as a separate entity.
 - When organizations are listed together at an address, expand any acronyms as a separate entity.
 - Identify any locations associated explicity associated with any of the potential organizations.
 - When organizations are listed together at a single address, ONLY associate the address with the last organization in the list.
 - Ignore any sub-units like departments or colleges.

### OUTPUT_FORMAT:
 - Output each JSON object on a separate line with no blank lines between them.
 - Replace any LaTeX escape sequences with utf-8 characters.
 - Use UTF-8 characters instead of Unicode escape sequences (e.g.: replace \u00e9 with é).
 - Do not add any extra text, explanation or annotation before or after the JSON objects.
 - Do not return a json Array.
 - double-escape all backslashes
 - Only report the main organizations like universities, universi, commissions, foundations or corporations.
 - Ignore sub-units like department, dipartimento, or college.
 - Do not include duplicate organizations.
 - Normalize organizations names to their most common full form.
 - Follow this pseudocode to generate the output:
```
    if no organizations are found, then output "null".
    else
        for each organization
            let org_name be the organization name.
            let city be "" unless you identied a city for this organization
            let country be "" unless you identified a country location for this organization
            output a json Object with this format: {{"name":org_name, "city":city , "country":country}}
```
### SOURCE_TEXT:
{input_text}\n\n
""".strip()


VERIFY_TEMPLATE = """
Match the institution names in the LIST_OF_NAMES with the contents of the SOURCE_TEXT.
Answer "True" if ALL the institutions in the LIST_OF_NAMES are present in the SOURCE_TEXT, otherwise answer "False"\n
Only respond with "True" or "False".
### LIST_OF_NAMES:\n
{inst_list}\n\n

### SOURCE_TEXT:\n
{source_text}\n\n
""".strip()

def find_included_files(wrapped_file):
    '''Search for inclusion macros'''
    include_pat = re.compile(r'\\(?:input|include|subfile)\s*(?:\[.+\])?\s*\{([^}]+)\}')
    matches = include_pat.findall(wrapped_file.read())
    if not matches:
        return []
    matches = [x.lstrip('./') for x in matches]
    matches = [
        x if x.endswith('.tex') else f"{x}.tex" 
        for x in matches
    ]
    return matches

def find_doc_class(wrapped_file, name_match=False, sub_match=False, auth_match=False):
    '''Search for document class related lines in a file  and return a code to represent the type'''
    doc_class_pat = re.compile(r"^\s*\\document(?:style|class)")
    sub_doc_class = re.compile(r"^\s*\\document(?:style|class).*(?:\{standalone\}|\{subfiles\})")

    for line in wrapped_file:
        if auth_match:
            # we can miss if there are two or more lines with documentclass
            # and the first one is not the one that has standalone/subfile
            if sub_doc_class.search(line):
                return -99999
            return 1.75 #main_files[tf] = 1
        if doc_class_pat.search(line):
            if name_match:
                # we can miss if there are two or more lines with documentclass
                # and the first one is not the one that has standalone/subfile
                if sub_doc_class.search(line):
                    return -99999
                return 1.5 #main_files[tf] = 1
            if sub_match:
                if sub_doc_class.search(line):
                    return -99999
                return 0.5
            return 1.0
    return -0.5 if sub_match else 0 #main_files[tf] = 0


def find_main_tex_source_in_tar(tar_bytes, encoding='utf-8', all_found=False, with_weights=False, file_path=None):
    '''Identify the main Tex file in a tarfile.

    Args:
        tar_bytes: A bytes from a gzipped file or a tar archive 
        of a directory containing tex source and support files.
    '''
    #auth_tex_names = set(["authlist", "author"])
    main_tex_names = set(["arxiv", "paper", "main", "ms.", "article", "manuscript", "neurips"])
    sub_tex_names = set(["appendix", "supplementary", "template"])

    tex_files = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r') as in_tar:
        tex_files = [f for f in in_tar.getnames() if f.endswith('.tex')]

        # got one file
        if len(tex_files) == 1:
            if with_weights:
                return ([(tex_files[0], 1),], None)
            else:
                return ([tex_files[0],], None)

        main_files = {}
        included_files = {}
        for tf in tex_files:
            depth = len(tf.split('/')) - 1
            has_auth_name = False #any(kw in tf.lower() for kw in auth_tex_names)
            has_main_name = any(kw in tf.lower() for kw in main_tex_names)
            has_sub_name = any(kw in tf.lower() for kw in sub_tex_names)
            try:
                fp = in_tar.extractfile(tf)
                wrapped_file = io.TextIOWrapper(fp, newline=None, encoding='utf-8') #universal newlines
                # does it have a doc class?
                # get the type
                main_files[tf] = find_doc_class(
                    wrapped_file,
                    name_match=has_main_name,
                    sub_match=has_sub_name,
                    auth_match=has_auth_name,
                ) - depth
                wrapped_file.close()
                # need a fresh fp for detection included files
                fp = in_tar.extractfile(tf)
                wrapped_file = io.TextIOWrapper(fp, newline=None, encoding='utf-8') #universal newlines
                included_files[tf] = [x for x in find_included_files(wrapped_file) if x in tex_files]
                wrapped_file.close() 
            except UnicodeDecodeError:
                try:
                    raw_data = in_tar.extractfile(tf).read() #peek(50000)
                    result = chardet.detect(raw_data)
                    detected_encoding = result["encoding"]
                    del raw_data
                    del result
                    fp = in_tar.extractfile(tf)
                    wrapped_file = io.TextIOWrapper(
                        fp, 
                        newline=None, 
                        encoding=detected_encoding, 
                        errors="replace"
                    ) #universal newlines
                    main_files[tf] = find_doc_class(
                        wrapped_file,
                        name_match=has_main_name,
                        sub_match=has_sub_name,
                    ) - depth
                    wrapped_file.close()
                    del fp
                    # need a fresh fp for detection included files
                    fp = in_tar.extractfile(tf)
                    wrapped_file = io.TextIOWrapper(fp, newline=None, encoding=detected_encoding, errors='replace') #universal newlines
                    included_files[tf] = [x for x in find_included_files(wrapped_file) if x in tex_files]
                    wrapped_file.close()
                    del fp
                except Exception as e:
                    print(
                        f"\nfind_main_tex_source_in_tar() failed to read {file_path}-{tf} with"
                        f" detected encoding {detected_encoding}: {e}"
                    )
                    for del_item in ['fp', 'raw_data', 'result']:
                        if del_item in locals():
                            del locals()[del_item]
                    raise e
        # return all if asked
        if all_found and with_weights:
            return (
                sorted(main_files.items(), key=lambda x: x[1], reverse=True),
                included_files
            )
        if all_found:
            return (
                sorted(main_files, key=main_files.get, reverse=True),
                included_files
            )

        # got one file with doc class
        if len(main_files) == 1:
            return (
                [main_files.keys()[0],],
                included_files
            )

        # account for multi-file submissions
        return (
            [max(main_files, key=main_files.get)],
            included_files
        )

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
    del tar_bytes
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

def append_node_contents(focus_nodes, full_nodelist, result_list, max_followers=3):
    for i,node in focus_nodes:
        temp_list = []
        temp_list.append(node.latex_verbatim())
        try:
            idx_plus = 1
            group_streak=False
            follow_count = 0
            while True:
                if idx_plus > 10:
                    break
                if follow_count > max_followers:
                    break
                if (not group_streak) and (len(temp_list) > 2):
                    break
                follow_node = full_nodelist[i+idx_plus]
                if isinstance(follow_node, LatexGroupNode):
                    temp_list.append(follow_node.latex_verbatim())
                    group_streak = True
                    follow_count += 1
                elif isinstance(follow_node, LatexCharsNode):
                    if not str(follow_node.chars).isspace():
                        group_streak = False
                else:
                    group_streak = False
                idx_plus += 1
        except IndexError:
            pass
        result_list.append("".join(temp_list))

def source_from_archive(tar_bytes, tex_main=None, file_path=None):
    ''' Get the source from tar or gz bytes copied from GCP.
        gz is assumed if tex_main is None
    '''
    #assert tex_main is not None, "tex_main is not defined"
    try:
        if tex_main:
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r') as in_file:
                fp = in_file.extractfile(tex_main)
                wrapped_file = io.TextIOWrapper(fp, newline=None, encoding='utf-8') #universal newlines
                source_text = wrapped_file.read()
                del fp
                del wrapped_file
        else:
            with gzip.open(filename=io.BytesIO(tar_bytes), mode='rt') as in_file:
                source_text = in_file.read()

    except UnicodeDecodeError:
        for del_item in ['fp', 'wrapped_file']:
            if del_item in locals(): del locals()[del_item]
        try:
            if tex_main:
                with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r') as in_file:
                    raw_data = in_file.extractfile(tex_main).read()
            else:
                with gzip.open(filename=io.BytesIO(tar_bytes), mode='rb') as in_file:
                    raw_data = in_file.read()
            result = chardet.detect(raw_data)
            detected_encoding = result["encoding"]
            del raw_data

            if tex_main:
                with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r') as in_file:
                    fp = in_file.extractfile(tex_main)
                    wrapped_file = io.TextIOWrapper(fp, newline=None, encoding=detected_encoding) #universal newlines
                    source_text = wrapped_file.read()
                    del fp
                    del wrapped_file
            else:
                with gzip.open(filename=io.BytesIO(tar_bytes), mode='rt', encoding=detected_encoding) as in_file:
                    source_text = in_file.read()

        except Exception as e:
            print(
                f"source_from_archive() failed to read {file_path}-{tex_main} with"
                f" detected encoding {detected_encoding}: {e}"
            )
            for del_item in ['fp', 'raw_data', 'wrapped_file', 'result']:
                if del_item in locals(): del locals()[del_item]
            return None  

    return source_text

        
def extract_pre_abstract_content(tar_bytes, tex_main=None, include_list=None, file_path=None, yield_sync=False):
    """
    Parses a .tex file:
    - Removes LaTeX comments
    - Extracts using latex macros
    - Extracts institution names (via recursive regex)
    - Extracts text before the abstract
    """
    auth_tex_names = set(["authlist", "author", "affil"])
    
    incl_res_gen_list = []
    if include_list:
        for inc_file in include_list:
            incl_res_gen_list.append(
                extract_pre_abstract_content(tar_bytes, tex_main=inc_file, file_path=file_path, yield_sync=True)
            )
    
    source_text = source_from_archive(tar_bytes, tex_main, file_path=file_path)
    if source_text is None:
        return None

    # Remove LaTeX comments (lines starting with non-escaped %)    
    new_def_v4 = r"""
    \\(newcommand|def|newcolumntype|renewcommand|providecommand|DeclareMathOperator|DeclareRobustCommand|newenvironment|renewenvironment|DeclareOption|newlength|newtheorem)\s*\{[^\}]+\}\s*(\[[^\]]*\])*(?P<brgrp>\s*\{\s*(?P<inner>(?>\s+|\\\\+|\\[{}]|[^{}\\]+|\\)+|\{(?P>inner)\}+)+\s*\})+
    """.strip()

    #old # \\newenvironment\{[^\}]+\}\s*\{\s*((?>[^{}]+|\{(?1)\})*)\}\s*\{\s*((?>[^{}]+|\{(?1)\})*)\}
    # old \\newenvironment\s*\{[^\}]+\}\s*(\[[^\]]*\])*(\{\s*(\s*(?>[^{}]+|\{(?3)\})*)+\}){1,3}
    #\\newenvironment\s*\{[^\}]+\}\s*(\s*\[[^\]]*\])*(         \s*\{\s*(      \s*(?>   (\\\\+|\\[{}]|[^{}\\]    )+|\{(?3)      \})*|\\)+\}                ){1,3}
    #\\newenvironment\s*\{[^\}]+\}\s*(\s*\[[^\]]*\])*(?P<brgrp>\s*\{\s*(?P<inner>(?>\s+|\\\\+|\\[{}]|[^{}\\]+|\\                 )+|\{(?P>inner)\}+)+\s*\}){1,3}

    strip_env = r"""
    \\newenvironment\s*\{[^\}]+\}\s*(\s*\[[^\]]*\])*(?P<brgrp>\s*\{\s*(?P<inner>(?>\s+|\\\\+|\\[{}]|[^{}\\]+|\\)+|\{(?P>inner)\}+)+\s*\})+
    """.strip()


    strip_provcmd = r"""
    \\providecommand\{[^\}]+\}\s*(\[[^\]]*\])?\s*\{\s*((?>(\\[{}]|[^{}])+|\{(?:[^{}]*|(?1))\})*)\}
    """.strip()

    new_def_v4_pat = re.compile(new_def_v4, re.DOTALL)
    strip_env_pat  = re.compile(strip_env, re.DOTALL)
    strip_provcmd_pat  = re.compile(strip_provcmd, re.DOTALL)

    content = re.sub(r"(?<!\\)%.*", "", source_text)
    content = strip_provcmd_pat.sub("\n", content)
    content = strip_env_pat.sub("\n", content)
    content = new_def_v4_pat.sub("\n", content)
    
    # try parsing latex:
    # Note: names are lowered before compare
    auth_macros = set([
        "author", "auth", "authors",
        "institute", "inst", "institution",
        "university",
        "orgname",
        "affiliation", "affil", "affiliations", "aff",
        "address",
        "cmsinstitute", "icmlaffiliation",
    ])
    supstr = set([
        "\\textsuperscript",
    ])
    latex_extracted_institutions = []

    
    try:
        lxwkr = LatexWalker(content, tolerant_parsing=True)
        (nodelist, pos, len_) = lxwkr.get_latex_nodes()
        focus_nodes = [
          (i,node) for i,node in enumerate(nodelist)
          if isinstance(node, LatexMacroNode) and node.macroname.lower() in auth_macros
        ]
        if focus_nodes:
            append_node_contents(focus_nodes, nodelist, latex_extracted_institutions)
            # Get /textsuperscript contents if indicated
            # @todo: also get the $^[1]$Institution style indicators 
            if any(pat in lx for lx in latex_extracted_institutions for pat in supstr):
                sup_res = extract_texsuperscript(nodelist)
                latex_extracted_institutions.extend(sup_res)
        else:
            doc = [
                node for node in nodelist
                if isinstance(node, LatexEnvironmentNode) and node.environmentname=='document'
            ]
            if doc:
                docnodelist = doc[0].nodelist
                focus_doc_nodes = [
                  (i,node) for i, node in enumerate(docnodelist)
                  if isinstance(node, LatexMacroNode) and node.macroname.lower() in auth_macros
                ]
                append_node_contents(focus_doc_nodes, docnodelist, latex_extracted_institutions)
                # Get /textsuperscript contents if indicated
                # @todo: also get the $^[1]$Institution style indicators 
                if any(pat in lx for lx in latex_extracted_institutions for pat in supstr):
                    sup_res = extract_texsuperscript(docnodelist)
                    latex_extracted_institutions.extend(sup_res)
        for var in ('nodelist', 'pos', 'len_', 'sup_res', 'focus_nodes', 'doc', 'docnodelist', 'focus_doc_nodes'):
            if var in locals(): del locals()[var]

    except Exception as e:
        print(f"\nOverly broad except in extract_pre_abstract_content(): {e} for {file_path}-{tex_main}")
        for var in ('nodelist', 'pos', 'len_', 'sup_res', 'focus_nodes', 'doc', 'docnodelist', 'focus_doc_nodes'):
            if var in locals(): del locals()[var]
        pass
    
    for inc_gen in incl_res_gen_list:
        inc_res = next(inc_gen)
        if inc_res:
            latex_extracted_institutions.append(inc_res)
    if latex_extracted_institutions or yield_sync:
        #res_list.append(latex_extracted_institutions)
        yield "\n".join(latex_extracted_institutions)
    
    #  "recursive" regex:
    #   ((?>[^{}]+|\{(?1)\})*)
    # optional brackets
    #   (:?\[\d+\])?\s*
    # This matches text possibly containing normal characters or nested braces,
    # until the outermost braces are matched.
    # If your LaTeX does not have deep nesting, this mainly ensures things like $^{1}$ are correctly parsed.
    
    # Remove LaTeX comments (lines starting with non-escaped %)
    
    institution_patterns = [
        r"\\affiliation\s*(?:\[\d+\])?\s*\{((?>[^{}]+|\{(?1)\})*)\}",
        r"\\institute\s*(?:\[\d+\])?\s*\{((?>[^{}]+|\{(?1)\})*)\}",
        r"\\address\s*(?:\[\d+\])?\s*\{((?>[^{}]+|\{(?1)\})*)\}",
        r"\\inst\s*(?:\[\d+\])?\s*\{((?>[^{}]+|\{(?1)\})*)\}",
        r"\\affil\s*(?:\[\d+\])?\s*\{((?>[^{}]+|\{(?1)\})*)\}",
        r"\\author\s*(?:\[\d+\])?\s*{[^}]+}{([^}]+)}",
        r"\\cmsinstitute\s*(?:\[\d+\])?\s*{[^}]+}{([^}]+)}",
        r"\\icmlaffiliation\s*(?:\[\d+\])?\s*{[^}]+}{([^}]+)}",
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
    for inc_gen in incl_res_gen_list:
        inc_res = next(inc_gen)
        if inc_res:
            extracted_institutions.append(inc_res)
    if extracted_institutions or yield_sync:
        # You can change the join method; here we join by newline and use set to deduplicate
        #return "\n".join(set(extracted_institutions))
        #res_list.append("\n".join(set(extracted_institutions)))
        yield "\n".join(set(extracted_institutions))

    # If no institution found, try extracting the text before the abstract
    text_extract_list = []
    for inc_gen in incl_res_gen_list:
        inc_res = next(inc_gen)
        if inc_res:
            text_extract_list.append(inc_res)
    
    match = re.split(
        r"\\begin\s*{\s*abstract\s*}|\\s*\\section\s*{\s*Abstract\s*}",
        content,
        maxsplit=1,
        flags=re.IGNORECASE
    )
    if (len(match) > 1):
        text_extract_list.append(match[0].strip())
        #return match[0].strip()
        #res_list.append(match[0].strip())
    if text_extract_list or yield_sync:
        yield "\n".join(text_extract_list)


    # If still not found, return the first 1/3 of the content as a fallback
    content_extract_list = []
    for inc_gen in incl_res_gen_list:
        inc_res = next(inc_gen)
        if inc_res:
            content_extract_list.append(inc_res)
    
    content_length = len(content)
    if content_length > 0:
        if tex_main and any(p in tex_main for p in auth_tex_names):
            content_extract_list.append(content.strip())
        else:
            one_third_length = max(content_length//3, 2000)
            #return content[:one_third_length].strip()
            #res_list.append(content[:one_third_length].strip())
            content_extract_list.append(content[:one_third_length].strip())
    if content_extract_list or yield_sync:
        yield "\n".join(content_extract_list)

    # If still not found, return an empty string
    #if res_list:
    #  yield res_list
    #else:
    # yield ["",]
    for i in reversed(range(len(incl_res_gen_list))):
        del incl_res_gen_list[i]
    del incl_res_gen_list
    return None


def utf8_from_textpath(txt_path):
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
    del txt_bytes    
    return file_contents


def extract_select_pages_from_txt(txt_path):
    """
    Given the gs bucket path to the plain text file, return the
    first 2 pages, then the second to last, and finally the last page.

    Args:
        file_contents (str): The full text content of the paper.

    Returns:
        list of page contents
    """
    file_contents = utf8_from_textpath(txt_path)

    # Split the text by form feed (page break)
    contents = file_contents.split("\u000C")
    del file_contents

    page_list = list(contents[0:2])
    if len(contents) >= 2:
        page_list.append(contents[-2])
    if len(contents) >= 1:
        page_list.append(contents[-1])
    del contents
    return page_list


def query_gemini_api(input_text, **kwargs):
    """
    Sends a request to the Gemini API to judge quality of result.
    """
    prompt = PROMPT_TEMPLATE.format(input_text=input_text)
    if not 'generation_config'in kwargs:
        kwargs['generation_config'] = GEN_CONFIG

    start_time = time.time()
    response = model.generate_content(prompt, **kwargs)
    end_time = time.time()

    timecost = end_time - start_time

    if response and response.text:
        clean_response = response.text
        # print(f"Execution time: {timecost:.4f} seconds")
        return clean_response
    else:
        print("\nAPI request failed or empty response")
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
        print("\nAPI request failed or empty response")
        return None

##########################
# Process one
# #########################
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
    '''Check sources in source list until one gives a good result
    '''
    width = 20000
    overlap = 500
    
    gemini_res = None
    break_outer = False
    for i,src in enumerate(src_list_gen):
        if len(src) < 30000:
            gemini_res = query_gemini_api(src)
        else:
            # overlapping windows:
            composite_res = []
            rng_srt = range(0,     len(src), width-overlap)
            rng_stp = range(width, len(src), width-overlap)
            for i, j in itr.zip_longest(rng_srt, rng_stp):
                composite_res.append(query_gemini_api(src[i:j]))
                if j is None: break
            gemini_res = '\n'.join(composite_res)
        if verbose:
            print(f"{i}: {src}\n")
            print(gemini_res)
        if gemini_res is None:
            continue
        for raw_row in gemini_res.strip().splitlines():
            if raw_row.startswith('`'):
                # we got a markdown "cell" from gemini
                continue
            if any((raw_row.startswith(pat)) for pat in BAD_START_PATTERNS):
                gemini_res = None
                break
            if any((pat in raw_row) for pat in BAD_IN_PATTERNS):
                gemini_res = None
                break
            else:
                break_outer = True
                break #from rows
        if break_outer:
            break # from src_list
    return gemini_res    


def is_good_result(res):
    if res is None:
        return False
    if res.startswith("null"):
        return False
    return True

def bytes_from_tarpath(tar_path):
    client = storage.Client(project=PRD_PROJECT)
    bucket = client.bucket(PRD_BUCKET_LOC)
    blob = bucket.blob(tar_path)
    try:
        tar_bytes = blob.download_as_bytes()
        return tar_bytes
    except (FileNotFoundError, ClientError, InvalidResponse, GoogleAPIError, NotFound, Forbidden) as e:
        raise e

def check_latex_with_gemini(arx_id, verbose=False):
    yymm = arx_id.split(".")[0]
    paper_id = arx_id.split("v")[0]
    tar_path = f"ftp/arxiv/papers/{yymm}/{paper_id}.tar.gz"

    res = None
    if verbose:
        print(f"Processing {tar_path}")
    try:
        tar_bytes = bytes_from_tarpath(tar_path)
        candidate_files, include_dict = find_main_tex_source_in_tar(tar_bytes, all_found=True, file_path=tar_path)
        for c_file in candidate_files:
            if verbose:
                print(f"\tProcessing {tar_path}, {c_file}")
            inc_list = include_dict.get(c_file, None) if isinstance(include_dict, dict) else None
            src_list_gen = extract_pre_abstract_content(tar_bytes, tex_main=c_file, include_list=inc_list, file_path=tar_path)
            res = check_src_list_with_gemini(src_list_gen, verbose=verbose)
            if is_good_result(res):
                del tar_bytes
                return res
    except (FileNotFoundError, ClientError, InvalidResponse, GoogleAPIError, NotFound, Forbidden) as e:
        # Probably single file latex or not a latex source paper
        if 'tar_bytes' in locals():
            del locals()['tar_bytes']
        pass
    ## Is it gz?
    gz_path = f"ftp/arxiv/papers/{yymm}/{paper_id}.gz"
    if verbose:
        print(f"Processing {gz_path}")
    try:
        gz_bytes = bytes_from_tarpath(gz_path)
        src_list_gen = extract_pre_abstract_content(gz_bytes, tex_main=None, file_path=tar_path)
        res = check_src_list_with_gemini(src_list_gen, verbose=verbose)
        del gz_bytes
        if is_good_result(res):
            return res
    except (ClientError, InvalidResponse, GoogleAPIError, NotFound, Forbidden):
        if 'gz_bytes' in locals():
            del locals()['gz_bytes']
        pass
    return "null"
    
def check_text_with_gemini(arx_id, verbose=False):
    yymm = arx_id.split(".")[0]
    paper_id = arx_id.split("v")[0]
    txt_path = f'txt/arxiv/{yymm}/{arx_id}.txt'

    res = None
    if verbose:
        print(f"Processing {txt_path}")
    try:
        src_list = extract_select_pages_from_txt(txt_path)
    except (FileNotFoundError, ClientError, GoogleAPIError, NotFound, Forbidden):
        src_list = []
        
    res = None
    for src in src_list:
        res = query_gemini_api(src)
        # res from trie
        if verbose:
            print(res)
        if res.startswith("null"):
            res = None
        else:
            break
    if (res is None):
        res = "null"
    return res

##########################
# Threaded processing for multiple files
# #########################
def get_single_file_results(arx_id, lock=None, pbar=None, verbose=False, vverbose=False):
    #paper_id = arx_id.split("v")[0]
    
    pid = os.getpid()
    if lock is not None:
        with lock:
            with open(f"logs/worker_process_{pid}.log", "a") as infile:
                infile.write(f"{arx_id} start\n")
                infile.flush()
                
    # Phase 1 - get names from text + Phase 2
    gemini_res = []
    latex_res = check_latex_with_gemini(arx_id, verbose=vverbose)
    if latex_res != "null":
        gemini_res.append(latex_res)
    else:
        text_res = check_text_with_gemini(arx_id, verbose=vverbose)
        if text_res != "null":
            gemini_res.append(text_res)
            
    gemini_res_str = '\n'.join(gemini_res)
    results = []
    found_institutions = False
    if gemini_res_str and gemini_res_str.strip().lower() != "null":
        # get extracted institutions
        institutions_found = []
        for raw_row in gemini_res_str.strip().splitlines():
            if raw_row.startswith('`'):
                # we got a markdown "cell" from gemini
                continue
            if (not raw_row) or (len(raw_row) < 5):
                continue
            row = raw_row.strip("[]").strip(',') # we might get a json list instead of line json
            if row:
                try:
                    institutions_found.append(json.loads(row))
                except json.JSONDecodeError as e:
                    try:
                        institutions_found.append(json.loads(r"{}".format(row).replace('\\', '\\\\')))
                    except json.JSONDecodeError:
                        if verbose:
                            print(f"JSONDecodeError: {e} on {arx_id} at {row}")
                        pass
        # get ROR
        for institution in institutions_found:
            try:
                clean_name = ""
                clean_city = ""
                clean_cntry = ""
                i_name = institution.get('name',"")
                i_city = institution.get('city',"")
                i_cntry = institution.get('country',"")
                if i_name:
                    clean_name = i_name.replace('{','').replace('}','').strip()
                if i_city:
                    clean_city = i_city.replace('{','').replace('}','').strip()
                if i_cntry:
                    clean_cntry = i_cntry.replace('{','').replace('}','').strip()
                if clean_name:
                    ror = ROR_FINDER.get_ror(
                        inst_name=clean_name,
                        inst_city=clean_city,
                        inst_cntry=clean_cntry,
                    )
                    if not ror in [x[3] for x in results]:
                        results.append((arx_id, clean_name, clean_city, ror))
                    found_institutions = True
            except Exception as e:
                if verbose:
                    print(f"Error: {e} on {arx_id} with {institution}")
                    print(repr(institution))
                pass
    if not found_institutions:
        results.append((arx_id, "null", "null", "null")) 
    
    if lock is not None:
        with lock:
            with open(f"logs/worker_process_{pid}.log", "a") as infile:
                infile.write(f"{arx_id} stop\n")
                
    return results

def process_tex_files(article_list, max_files=None, max_workers=5, verbose=False):
    start_time = time.time()
    if max_files:
        article_list = article_list[:max_files]
    estimated_tex_files = len(article_list)

    results = []
    lock = threading.Lock()
    
    pid = os.getpid()
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/worker_process_{pid}.log", "w") as infile:
        pass

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #, tqdm(total=estimated_tex_files, desc="Processing .tex files") as pbar:
        future_to_article = {
            executor.submit(get_single_file_results, arx_id, lock=lock, pbar=None): str(arx_id)
            for arx_id in article_list
        }
        successes = []
        try:
            for future in as_completed(future_to_article, timeout=240):
                article = future_to_article[future]
                try:
                    article_results = future.result()
                    results.extend(article_results)
                    successes.append(article)
                except Exception as e:
                    if verbose: 
                        print(f"\n❌ Error processing article '{article}': {e}")
                    results.append((article, 'error', f"{e}", 'null'))
                    #time.sleep(.5)
                    # try:
                    #     article_results = get_single_file_results(article)
                    #     results.extend(article_results)
                    #     successes.append(article)
                    # except Exception as e2:
                    #     if verbose: 
                    #         print(f"❌ Error processing article '{article}': {e}")
                    #     results.append((article, 'error'))
        except TimeoutError as e_time:
            failures = [
                a_id for a_id in future_to_article.values()
                if not a_id in set(successes)
            ]
            for article in failures:
                results.append((article, 'error', 'timeout', 'null'))
                if verbose:
                    print(f"\n❌ Error processing article '{failures}': {e_time}")

    if verbose:
        total_time = time.time() - start_time
        print(f"✅ Total processing time: {total_time:.2f} seconds")
    gc.collect()
    return results

############################
# # Phase 2
# ############################
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
    
    def __init__(self, prompt_template=None, doc_k=5, RECREATE_INDEX=False):
        if prompt_template is None:
            self.prompt_template = ROR_TEMPLATE
        else:
            self.prompt_template = prompt_template
        self.doc_k = doc_k
        self.RECREATE_INDEX = RECREATE_INDEX
        self.ror_cache = {}
        self.ror_gspath = 'gs://institutional-extract-scratch/reference/v1.63-2025-04-03-ror-data_schema_v2.json'
        self.model_project = 'arxiv-development'
        self.model_bucket_loc = 'institutional-extract-scratch'
        self.dest_blob_name = "models/ror_index_city_and_noncity_abbrev_county_withdrawn.zip"
        self.local_index = "ror_index_city_and_noncity_abbrev_county_withdrawn"
        self.withdrawn_map = None
        self.qa_chain = self.build_qa_chain()
        
        
    def build_withdrawn_map(self):
        fs = gcsfs.GCSFileSystem()
        with fs.open(self.ror_gspath, "r", encoding="utf-8") as f:
            ror_data = json.load(f)

        #Locate withdrawn and successors
        ror_dict = {e['id']:e for e in ror_data}
        wd_succ_dict = {}
        inactive_set = set([
            'withdrawn',
            'inactive',
        ])
        withdrawn_ror = {
            e['id']: e['id']
            for e in ror_data 
            if e.get('status',"") in inactive_set
        }
        max_follows = 10
        follow_count = 0
        while len(withdrawn_ror) > 0:
            if follow_count > max_follows:
                break
            for wd_ror, sc_ror in tqdm(withdrawn_ror.items()):
                wd_entity = ror_dict[sc_ror]
                successor_rel = [
                    r['id'] for r in wd_entity.get('relationships',[])
                    if r['type'] == 'successor'
                ]
                if len(successor_rel) < 1:
                    continue
                succ_ror = successor_rel[0]
                wd_succ_dict[wd_ror] = succ_ror
            withdrawn_ror = {
                wd_ror: sc_ror 
                for wd_ror, sc_ror in wd_succ_dict.items()
                if sc_ror in withdrawn_ror.keys()
            }
            follow_count += 1
        self.withdrawn_map = wd_succ_dict
        return wd_succ_dict

    def build_qa_chain(self):
        # Load the index model, training it if needed.
        model_project    = self.model_project   
        model_bucket_loc = self.model_bucket_loc
        dest_blob_name   = self.dest_blob_name  
        local_index      = self.local_index     
        #os.chdir("/home/jupyter/metadata-vertexai/")

        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        client = storage.Client(project=model_project)
        bucket = client.bucket(model_bucket_loc)
        blob = bucket.blob(dest_blob_name)

        if self.RECREATE_INDEX or (not blob.exists()):
            fs = gcsfs.GCSFileSystem()
            with fs.open(self.ror_gspath, "r", encoding="utf-8") as f:
                ror_data = json.load(f)
                
            #Locate withdrawn and successors
            if self.withdrawn_map is None:
                self.withdrawn_map = self.build_withdrawn_map()
            wd_succ_dict = self.withdrawn_map
                
            #Parse into training docs
            docs = []
            docs = load_special_cases_ror()
            for i,entry in tqdm(enumerate(ror_data)):
                ror_id = entry.get("id", "")
                if not ror_id:
                    continue
                # Check if withdrawn and link to active successor
                ror_id = wd_succ_dict.get(ror_id, ror_id)
                
                locs = entry.get('locations',[])
                loc_name = ""
                ctry_name = ""
                try:
                    loc_name = f", {locs[0]['geonames_details']['name']}"
                except KeyError:
                    pass
                try:
                    ctry_name = f", {locs[0]['geonames_details']['country_name']}"
                except KeyError:
                    pass
                for name_info in entry.get("names", {}):
                    name = name_info.get("value", "")
                    if not name:
                        continue
                    if ctry_name and 'acronym' in name_info.get("types", []):
                        content = f"{name}{ctry_name} — {ror_id}"
                        docs.append(Document(page_content=content))
                    else: 
                        no_loc_content = f"{name} — {ror_id}"
                        docs.append(Document(page_content=no_loc_content))
                    if loc_name:
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

        prompt = PromptTemplate(input_variables=["question", "context"], template=self.prompt_template)

        #  Build a Retrieval + QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": self.doc_k}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        return qa_chain

    @ft.lru_cache(maxsize=1000)
    def qa_chain_invoke(self, inst_str):
        return self.qa_chain.invoke({"query": inst_str})
    
    #@ft.cache # tends to cache issues too, so we rolled out own
    def get_ror(self, inst_name, inst_city="", inst_cntry=""):
        # cache_lookup = self.ror_cache.get(inst_name)
        # if cache_lookup:
        #     return cache_lookup
        try:
            inst_loc = inst_city
            cap_count = sum(x.isupper() for x in inst_name)
            if len(inst_name) <= 3:
                inst_loc = inst_cntry
            if cap_count/len(inst_name) > .5:
                inst_loc = inst_cntry

            name_loc = f"{inst_name}, {inst_loc}"
            if not inst_loc:
                name_loc = inst_name
            response = self.qa_chain_invoke(name_loc)
            ror_id = response["result"].strip()
            if ror_id == 'null' and inst_loc:
                #try without loc
                response = self.qa_chain_invoke(inst_name)
                ror_id = response["result"].strip()
            #if ror_id != 'null':
            #    self.ror_cache[inst_name] = ror_id

        except Exception as e:
            print(f"Error querying {inst_name}: {e}")
            ror_id = "error"
        return ror_id.strip()
    
ROR_FINDER = rorFinder()
