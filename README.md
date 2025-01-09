# RAG_QA_Application

## Project Overview

Retrieval augmented generation or RAG, is an architectural approach that can improve the efficiency of large language model (LLM) applications by leveraging custom data. This is done by retrieving data/documents relevant to a question or task and providing them as context for the LLM.

### Technology
1. Python
2. LLM - Google Gemini Pro
3. LangChain
4. VectorDB- Chroma
5. Github
6. Embedding Model - GoogleGenerativeAIEmbeddings

## How to run?

#### Step 01 - Clone the repository
```bash
Project repo: https://github.com/
```

### Step 02 - Create a conda environment after opening the repository in git bash or terminal

```bash
conda create -n ragapps python=3.11 -y
```

```bash
conda activate ragapps
```

### Step 03 - Install all requirements
```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your Google API keys as follow:

```ini
GOOGLE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Step 04 - Run an application.

``` bash
python app.py
```



