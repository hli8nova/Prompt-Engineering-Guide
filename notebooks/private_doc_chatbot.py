import re
from io import BytesIO
from typing import Dict, Any, List
import databutton


# Modules to Import
import streamlit as st
from langchain import LLMChain, OpenAI
# from langchain.llms import OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
# from langchain.vectorstores import Chroma
from pypdf import PdfReader
from langchain.vectorstores import DeepLake

ACTIVELOOP_TOKEN = databutton.secrets.get("ACTIVELOOP_TOKEN")
OPENAI_API_KEY = databutton.secrets.get("OPENAI_API_KEY")
# !activeloop login --token ACTIVELOOP_TOKEN

@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)
        output.append(text)
    return output

def parse_txt(file: BytesIO) -> List[str]:

    # with open(file) as f:
    #     contract = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # pages = text_splitter.split_text(contract)
    pages = text_splitter.split_text(file)
    texts = text_splitter.create_documents(pages)

    return texts

@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            separators=["\n\n", "\n", ".", "!", "?"],
            chunk_overlap=100,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


# @st.cache_data
@st.cache_resource
def test_embed():
    embeddings = OpenAIEmbeddings(openai_api_key=api)
    # Indexing
    # Save in a Vector DB
    DB_TYPE = 'DEEPLAKE'
    with st.spinner("It is indexing..."):
        if DB_TYPE == 'FAISS':
            index = FAISS.from_documents(pages, embeddings)
        elif DB_TYPE == 'CHROMA':
            db_name = "my_contract"
            index = Chroma.from_documents(pages, embeddings, db_name)
        elif DB_TYPE == 'DEEPLAKE':
            org = "will-nova"
            dataset_path = "hub://" + org + "/my_contract2"
            index = DeepLake.from_documents(pages, 
                embeddings, 
                dataset_path=dataset_path, 
                token = ACTIVELOOP_TOKEN,
                overwrite=True
            )

    st.success("Embeddings done.", icon="✅")
    return index



st.title("🤖 Private PDF ChatBot")

st.sidebar.markdown(
    """
    ### Steps:
    1. Upload a File
    2. Perform Q&A

    **Note : File content and API key not stored in any form.**
    """
)
uploaded_file = st.file_uploader("**Upload Your Text File**", type=["txt"])
st.info(uploaded_file)

if uploaded_file:
    # name_of_file = uploaded_file.name
    # with open(name_of_file) as f:
    #     contract = f.read()
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    # pages = text_splitter.split_text(name_of_file)
    # texts = text_splitter.create_documents(pages)
    print(uploaded_file)
    # # texts = parse_txt(uploaded_file)
    # pages = text_to_docs(uploaded_file)
    # pages
    if False:
        # print(pages)
        with st.expander("Show Page Content", expanded=False):
            page_sel = st.number_input(
                label="Select Page", min_value=1, max_value=len(pages), step=1
            )
            pages[page_sel - 1]

        api = OPENAI_API_KEY
        if api:
            embeddings = OpenAIEmbeddings(openai_api_key=api)
            db = FAISS.from_documents(pages, embeddings)

            retriever = db.as_retriever()
            retriever.search_kwargs['distance_metric'] = 'cos'
            retriever.search_kwargs['k'] = 2

            query = st.text_input(
                "**What's on your mind?**",
                placeholder="Ask me anything from {}".format(name_of_file),
            )

            if query:
                with st.spinner(
                    "Generating Answer to your Query : `{}` ".format(query)
                ):
                    MODEL = "gpt-3.5-turbo" 
                    qa = RetrievalQA.from_chain_type(
                        llm=OpenAI(model_name=MODEL, openai_api_key=api), 
                        chain_type="stuff", 
                        retriever=retriever, 
                        return_source_documents=True
                    )
                    result = qa({"query": query})
                    # result = qa.run(query)
                    answer = result["result"]
                    source_docs = result["source_documents"]

                    # res = {"answer": answer, "source_docs": source_docs[:2]}
                    # res = qa_with_source(query=query, retriever=retriever)
                    st.info(answer, icon="🤖")
                    # st.info(body=('Source 1 Page ' + str(source_docs[0].metadata["page"]), '\n', str(source_docs[0].page_content)), icon="🔍")

                    for i, src in enumerate(source_docs):
                        st.info('Source ' + str(i+1) + ': ' + str(src.page_content) + '\n') #, icon="🔍")

            with st.expander("History/Memory"):
                st.session_state.memory


