import streamlit as st

from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

st.title("RAG Chatbot")

# === Set up model and prompt template ===
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=st.secrets["GOOGLE_API_KEY"], temperature=0.7)

system_prompt = "你在 Rasmus 同學的個人網頁上工作，提供面向人資的問答服務。你會收到**Context**字段，這是從王睿洋同學的資料庫中檢索到的相關內容。請根據這些內容回答問題。如果題目涉及上下文、缺乏資料或你不知道答案，請提示用戶開啟 RAG 功能"
prompt_template_with_rag = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "**Human Message:**{input}\n\n**Content:**{context}") 
    ]
)

prompt_template_without_rag = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "**Human Message:**{input}")
    ]
)

# === Set up Vector Store ===
@st.cache_resource
def get_vector_store():
    return Chroma(
        collection_name="my_database", 
        persist_directory="chroma", 
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"])
    )

vector_store = get_vector_store()
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# === Set up Retrieval Chain ===
retrieval_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=create_stuff_documents_chain(llm=model, prompt=prompt_template_with_rag)
)

# === Set up Message History ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

chat_with_rag = RunnableWithMessageHistory(
    retrieval_chain,
    lambda session_id: st.session_state.chat_history,  # Lambda to retrieve history from session_state
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

chat_without_rag = RunnableWithMessageHistory(
    prompt_template_without_rag | model,
    lambda session_id: st.session_state.chat_history,  # Lambda to retrieve history from session_state
    input_messages_key="input",
    history_messages_key="chat_history",
)

CONFIG = {"configurable": {"session_id": "streamlit_chat_session"}}  # st.session_state is unique to each user's browser session.

# === Streamlit Chat UI ===
if "messages" not in st.session_state:
    st.session_state.messages = []

option_map = {
    0: ":material/database: RAG",
}
selection: list = st.segmented_control( 
    "**Tool**",
    options=option_map.keys(),
    format_func=lambda option: option_map[option],
    selection_mode="single",
)

# Display previous messages from the session state.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("How can I help you today?")

# Handle user input from the chat input box.
if prompt:
    # Add user message to the display list and show it.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the assistant's response.
    with st.chat_message("assistant"):
        if selection == 0:
            stream_generator = chat_with_rag.stream(
                {"input": prompt},
                config=CONFIG
            )

            full_response = ""
            placeholder = st.empty()

            got_context = False
            for chunk in stream_generator:
                if "context" in chunk and not got_context:
                    context = chunk["context"]
                    got_context = True

                if "answer" in chunk:
                    full_response += chunk["answer"]
                    placeholder.markdown(full_response)
            
            if got_context:
                with st.expander("🔎 Retrieved Context", expanded=False):
                    for doc in context:
                        st.markdown(f"**Source:** {doc.metadata['source']} - Chunk {doc.metadata['chunk_index']}")
                        st.markdown(doc.page_content)
                        st.markdown("---")
        else:
            stream_generator = chat_without_rag.stream(
                {"input": prompt},
                config=CONFIG
            )

            full_response = st.write_stream(stream_generator)

    # Append the complete assistant's response to the display list.
    st.session_state.messages.append({"role": "assistant", "content": full_response})