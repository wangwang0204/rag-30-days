import streamlit as st

from langchain_chroma import Chroma

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

st.title("RAG Chatbot")
api_key = "AIzaSyCqlp4bV1ybgaNotfZFdscWa-Cu0x7pJ3o"

# === Set up model and prompt template ===
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0.7)
prompt_template_with_rag = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Provide concise and informative answers."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "**Human Message:**{input}\n\n**Content:**{context}") 
    ]
)

prompt_template_without_rag = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI assistant. Provide concise and informative answers."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "**Human Message:**{input}")
    ]
)

# === Set up Vector Store ===
vector_store = Chroma(collection_name="foo", persist_directory="chroma_db", embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key))
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

            for chunk in stream_generator:
                if "answer" in chunk:
                    full_response += chunk["answer"]
                    placeholder.markdown(full_response)
        else:
            stream_generator = chat_without_rag.stream(
                {"input": prompt},
                config=CONFIG
            )

            full_response = st.write_stream(stream_generator)

    # Append the complete assistant's response to the display list.
    st.session_state.messages.append({"role": "assistant", "content": full_response})