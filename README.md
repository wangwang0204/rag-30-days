# rag-30-days

## Abstract
《RAG 論文學習筆記 × 個人助手實作》系列的第一週，我們會介紹並使用 LangChain, FAISS, Streamlit 等工具搭建一個 RAG 系統。在學習與工作中，我們一定會累積大量的文本素材，例如履歷、作品集與各式各樣的申請書，如果一個 AI 系統能精確地從資料庫檢索我們寫過的素材，再進行生成，那對重複性高的工作申請、社團申請會方便許多。除此之外，我們也可以將系統部署在個人網站上，實現互動式、問答式的呈現。本系列的後半，會專注在RAG 領域重要文獻的閱讀，例如 DPR、HyDE 和 Agentic RAG、也會花一些篇幅探討現有的成熟應用以及圍繞 RAG 的實務議題。

## About This Repo 
*這個資料夾包含本系列前七天的內容，以下是簡介與使用說明。*

`notebooks` 這個資料夾包含了所有的 Jupyter Notebook 檔案，這些檔案是本系列教學的主要內容。每個 Notebook 都會詳細說明 RAG 系統的各個組件，以及如何使用它們來構建自己的應用。其中 `day4-create_db.ipynb` 會讀取 `raw_database` 並建立一個向量資料庫 -- `vector_store/`。讀者可以換成自己的資料集。

`demo.py` 是一個簡單的示範應用，展示了如何使用本系列教學中所介紹的技術來構建一個 RAG 系統。讀者可以參考這個範例，並根據自己的需求進行修改和擴展。

`raw_database` 是一個包含原始文本資料的資料夾，目前提供的 **chunking** 方法範例能處理 `.txt`, `.pdf`, `.docx`, `.md` 和 `.json` 格式的文件。讀者可以替換成自己的資料集，並在 `day4-create_db.ipynb` 中進行處理。

### API KEY 設置
- **notebooks**：google-genai 會自動讀取 `.env` 檔案中的 GOOGLE_API_KEY，也可以透過 `google_api_key` 參數設定。
- **demo.py**：在本地運行的時候，除了透過 `.env` ，也可將 GOOGLE_API_KEY 設置在 `.streamlit/secrets.toml` 中，並使用 `st.secrets["GOOGLE_API_KEY"]` 來獲取 api key。推薦使用後者（即目前的代碼），因爲使用 streamlit cloud 的環境變量也是使用一樣的讀取方式。

### 環境配置
- python 3.11
- requirements.txt

**參考流程**：

1. 下載此檔案夾

```
git clone https://github.com/wangwang0204/rag-30-days.git
```

2. 建立虛擬環境
```
conda create -n <name> python=3.11
conda activate <name>  # 啟動虛擬環境
```

3. 下載 dependencies（建議使用提供的版本）
```
pip install -r requirements.txt 
```

4. 建立 jupyter kernel
```
python -m ipykernel install --user --display-name "<Display Name>"
```
或
```
python3 -m ipykernel install --user --display-name "<Display Name>"
```

### 運行方式
```
streamlit run demo.py # 需要先建立 vector store
```

## Links
- [RAG 個人助手 Demo](https://personal-database-rag.streamlit.app)
- [Dataset, Kaggle](https://www.kaggle.com/datasets/leowang0204/simulated-personal-database-raw-data)