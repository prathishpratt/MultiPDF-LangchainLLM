import streamlit as st            #For GUI
from dotenv import load_dotenv
from PyPDF2 import PdfReader      #To read the input pdfs
from langchain.text_splitter import CharacterTextSplitter #To divide the text into chunks
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS          #Vector store to save the embedded chunks locally
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory  #To give memory to the model
from langchain.chains import ConversationalRetrievalChain
from htmlTemp import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    """
        To take the list of pdf files from input.
        Returns a single string of text with all the content of the pdfs
        Loops thro each pdf page by page and append to the string 'text'
    """

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
        Take that single string as input
        Return a list of chunk of text that we will feed into the vector DB  
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",                    #To separate
        chunk_size=1000,                   #For 1000 characters per chunk
        chunk_overlap=200,                 #To not end our chunk abrubly 
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks                          #A list of chunks of 1000 character size each


def get_vectorstore(text_chunks):
    """
        I used the free Instructor Xl model for the embedding.
        Apparently it is faster tha OpenAI's but since it will be trained locally,
        It will take a long time to create the embedding. 

        FAISS is to store those vectors and it will take the input of both the
        embedding and the original chunks. 
    """
    #embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """
        Takes the vector store as the input.
        We use langchain's ConversationBufferMemory for the memory of the model
        Memory here means that the model can remember the last question while answering next one.
        It takes the history of the convo and returns the next element of the convo.
    """

    #llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



#Streamlit might refreshes the program, so to not generate again, 
#we use the "st.session_state"
#So if it refreshes, it will check if "response" is there, if yes then nothing, else None
def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    #Loop over the entire chat history with the index and its content
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()   #Gets the api key from .env file
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)  #Loops the pdfs and returns a single string

                # get the text chunks
                text_chunks = get_text_chunks(raw_text) #Split the single string into chunks to be fed 

                # create vector store using Instructor Xl free
                vectorstore = get_vectorstore(text_chunks)  #To store the vector embeddings

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()