import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from templates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import openai

def get_text_from_pdf(pdfs):
    ## empty string(variable) to store all the text:
    text = ""
    ## Iterating through all user enterd pdfs:
    for pdf in pdfs:
        ## EXTRACTS PAGES FROM THE PDFS:
        pdf_reader = PdfReader(pdf)
        ## NOW LOOPING THROUGH PAGES:
        for page in pdf_reader.pages:
            ## FUNCTION TO EXTRACT TEXT FROM EACH PAGES
            text += page.extract_text()

    return text

## FUNCTION TO CONVERT ENTIRE RAW TEXT INTO SMALL CHUNKS OF TEXT
def get_text_chunks_raw(text):
    ## LANGCHAIN TEXT SPLITTER:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        ## MAX CHUNK OF SIZE CREATED 1000
        chunk_size = 1000,
        ## IF SOME INFORMATION IS EXTRACTED AND MISSED BETWEEN 1000 CHUNKS WE MOVE BACK 200 WORDS(OVERLAP BACKWARDS)
        chunk_overlap = 200,
        length_function = len
    )

    chunks = text_splitter.split_text(text)
    return chunks

## CONVERT CHUNKS TO VECTORS(EMBEDDINGS) AND STORING TO DATABASE:
def store_vector_embeddings(get_text_chunks):
    ## USING OPENAI EMBEDDINGS , WHICH CALLS THE OPENAI SERVERS -> FAST , DOSENT STORES ON LOCAL MACHINE
    embeddings = OpenAIEmbeddings()
    ## HUGGING FACE EMBEDDINGS , STORES THE EMBEDDINGS IN OUR LOCAL MACHINE -> SLOW BUT MORE ACCURATE
    ## new_embeddings = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-xl')
    ## STORING THE  IN FAISS STORAGE
    vectorstore = FAISS.from_texts(texts=get_text_chunks,embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        ## STORING THE EMBEDDINGS AS VECTOR
        retriever = vectorstore.as_retriever(),
        ## CREATING A BUFFER MEMORY
        memory = memory
    )


    return conversation_chain

## HANDLING USER INPUT AND SENDING THE RESPONSE:
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():

    ## TO ACCESS THE API-TOKENS:
    ## This loads variables from your .env file into the environment
    load_dotenv()
    openai_api_key = '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'
    openai.api_key = openai_api_key
    st.set_page_config(
        page_title="Chatting with Multipe Pdf's",
        page_icon=":books:",
        layout="wide",
        initial_sidebar_state="expanded"

    )

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Start Chatting with your PDFs :books:")
    user_question = st.text_input(":blue[Ask any Questions about your Uploaded Documents:]")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        ## STORING ALL THE INPUT PDFS IN VARIABLE:
        pdfs = st.file_uploader(":blue[Upload all Your PDFs here and click on 'Process']",accept_multiple_files=True)
        ## BUTTON TO PROCESS THE INPUT QUESTION
        if st.button("Process"):
            with st.spinner("Processing"):

                ## GET THE TEXT FROM PDF
                get_raw_text = get_text_from_pdf(pdfs)

                ## CONVERT IT INTO SMALL CHUNKS by CALLING 'get_text_chunks_raw' -> TAKES INPUT THE RAW TEXT AND RETURNS CHUNKS OF IT:
                get_text_chunks = get_text_chunks_raw(get_raw_text)

                ## CREATE EMBEDIINGS(REPRESENT CHUNKS INTO VECTOR) AND STORE IT IN DATABASE AS VECTORS:
                vectorstore = store_vector_embeddings(get_text_chunks)

                ##CREATEING CONVERSATION CHAIN:
                st.session_state.conversation = get_conversation_chain(vectorstore)

 
    
if __name__ == "__main__":
    main()