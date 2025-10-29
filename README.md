# rag-30-days

## Abstract
本專案展示了一個功能完善的 RAG (Retrieval-Augmented Generation) 系統的建構過程。專案旨在介紹並使用 LangChain, FAISS, Streamlit 等工具搭建一個 RAG 系統，並透過 10 篇部落格文章詳細說明其核心概念、技術與實作細節。透過本專案，讀者可以學習如何從零開始構建一個 RAG 應用，並將其應用於個人資料管理或互動式問答系統。

## 核心技術

本專案主要使用了以下技術和框架：

*   **RAG 系統**：結合檢索與生成，提升大型語言模型回答的準確性和相關性。
*   **LangChain**：一個強大的框架，用於開發由語言模型驅動的應用程式，簡化了 RAG 系統的建構流程。
*   **FAISS**：用於高效相似性搜索的函式庫，在本專案中用於向量資料庫的建立與檢索。
*   **Streamlit**：一個用於快速建立資料科學和機器學習應用程式的框架，用於構建使用者介面。

## 專案內容概覽


1.  RAG 系統的基本概念與運作原理。
2.  如何使用 LangChain 整合不同的語言模型組件。
3.  如何建立和管理向量資料庫 (Vector Store)，並利用 FAISS 進行高效檢索。
4.  如何使用 Streamlit 搭建一個互動式的 RAG 應用程式。

## About This Repo 

`notebooks` 這個資料夾包含了所有的 Jupyter Notebook 檔案，這些檔案是本系列教學的主要內容。每個 Notebook 都會詳細說明 RAG 系統的各個組件，以及如何使用它們來構建自己的應用。其中 `create_db.ipynb` 會讀取 `raw_database` 並建立一個向量資料庫 -- `vector_store/`（已在 repo 裡，streamlit app 需要讀取）。讀者可以換成自己的資料集。

`demo.py` 是一個簡單的示範應用，展示了如何使用本系列教學中所介紹的技術來構建一個 RAG 系統。讀者可以參考這個範例，並根據自己的需求進行修改和擴展。

`raw_database` 是一個包含原始文本資料的資料夾，目前提供的 **chunking** 方法範例能處理 `.txt`, `.pdf`, `.docx`, `.md` 和 `.json` 格式的文件。讀者可以替換成自己的資料集，並在 `create_db.ipynb` 中進行處理。

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

5. 加入 `.env` 和 `.streamlit./secrets.toml`

### 運行方式
```
streamlit run demo.py # 需要先建立 vector store
```

## Blogs

這個專案包含了 10 篇部落格文章，詳細介紹了 RAG 系統的建構過程、LangChain 的使用、FAISS 和 Streamlit 的實作細節。這些文章位於 `blogs/` 目錄下：

*   `1.Introduction.pdf`: 專案背景、目標以及 RAG 系統的基本概念。
*   `2.RAG_intro.pdf`: 深入探討 RAG (Retrieval-Augmented Generation) 系統的原理、架構與優勢。
*   `3.LangChain(1).pdf`: LangChain 系列第一部分，介紹核心組件及其在 RAG 系統中的應用。
*   `4.LangChain(2).pdf`: LangChain 系列第二部分，進階說明構建複雜語言模型應用的用法。
*   `5.LangChain(3).pdf`: LangChain 系列第三部分，涵蓋更多實用功能與最佳實踐。
*   `6.FAISS.pdf`: 詳細介紹 FAISS 函式庫，包括安裝、使用方法及如何在 RAG 系統中建立高效向量資料庫。
*   `7.RAG_system.pdf`: 整合前面知識，詳細說明如何從頭到尾建構一個完整的 RAG 系統。
*   `8.Streamlit.pdf`: 介紹如何使用 Streamlit 框架為 RAG 系統建立互動式網頁應用介面。
*   `9.Demo.pdf`: 展示 RAG 系統實際運作範例，引導讀者運行和體驗應用。
*   `10.Summary.pdf`: 總結專案學習內容，展望 RAG 技術的未來發展與應用。

## Links
- [RAG 個人助手 Demo](https://personal-database-rag.streamlit.app)
- [Dataset, Kaggle](https://www.kaggle.com/datasets/leowang0204/simulated-personal-database-raw-data)

*原本是 2025 IT 邦幫忙鐵人賽 -- 'RAG 論文學習筆記 × 個人助手實作' 的參賽作品，但我只寫了前十篇🤗，放在 `\blogs`*
