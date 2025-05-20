
## 🚀 New Features

### 1. Trie-Based Affiliation Extractor
- [`trie_extractor.py`](./trie_extractor.py)  
- Supporting helpers in the `utils/` package  
- **Build or Load**:  
  You can build the trie from scratch, or load a pre-built extractor:
  ```python
  import pickle
  from fsspec import filesystem

  fs = filesystem("gs")
  with fs.open("gs://institutional-extract-scratch/models/trie_extractor.pkl", "rb") as f:
      extractor = pickle.load(f)
### 2. `check_text_with_gemini` function
- **The trie extractor can now be combined with LLM extractor**:
  ```python
  def check_text_with_gemini(arx_id, verbose=False):
      """
      For a given arXiv ID:
        - run Gemini on selected pages;
        - fall back to 'null' if no result;
        - extract affiliations via a trie;
        - merge both sources into one de-duplicated JSON-lines string.

      Returns:
          str: newline-separated JSON objects, e.g.:
            {"name":"…","city":"…","country":"…"}
            {"name":"…","city":"…","country":"…"}
      """
