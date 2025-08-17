import streamlit as st
import uuid
import time

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

st.title("RAG Chatbot")

# === Set up model and prompt template ===
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=st.secrets["GOOGLE_API_KEY"], temperature=0.7)

system_prompt = \
    """
    你在 Rasmus 同學的個人網頁上工作，提供問答服務。問題的範圍包含 Rasmus 本人的學術資料，個人經驗與專案等等。
    請根據這些內容回答問題收到 **Context**字段，這是從 Rasmus 同學的資料庫中檢索到的相關內容。
    如果你未收到**Context**字段並且無法回答問題，請說：「我無法回答您的問題，請開啟 RAG 功能，以檢索關於 Rasmus 的相關資訊。」
    """.strip()
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
    return FAISS.load_local(
        "vector_store",
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"]),
        allow_dangerous_deserialization=True
    )
        
vector_store = get_vector_store()
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# === Set up Retrieval Chain ===
retrieval_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=create_stuff_documents_chain(llm=model, prompt=prompt_template_with_rag)
)

# === Set up Message History ===
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

def get_chat_history(session_id):
    if session_id not in st.session_state.chat_histories:
        st.session_state.chat_histories[session_id] = ChatMessageHistory()
    return st.session_state.chat_histories[session_id]

chat_with_rag = RunnableWithMessageHistory(
    retrieval_chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
)

chat_without_rag = RunnableWithMessageHistory(
    prompt_template_without_rag | model,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

CONFIG = {"configurable": {"session_id": st.session_state.session_id}}

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

def greeting_message():
    message = "你好！我是 Rasmus 的個人助手，有什麼我可以幫忙的嗎？"
    for s in message:
        time.sleep(0.01)  # Simulate typing delay
        yield s

if "greeting" not in st.session_state:
    st.session_state.greeting = True
    
with st.chat_message("assistant"):
    if st.session_state.greeting:
        st.write_stream(greeting_message)
        st.session_state.greeting = False
    else:
        st.markdown("你好！我是 Rasmus 的個人助手，有什麼我可以幫忙的嗎？")


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

            # context expander
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