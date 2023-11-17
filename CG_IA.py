import streamlit as st
from dotenv import load_dotenv
import dill as pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

load_dotenv()

def main():
    st.header("Chat with Genius")
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]
        vectorstore = None

        if os.path.exists(f"{store_name}.pkl"):
            file_size = os.path.getsize(f"{store_name}.pkl")
            st.write(f"File size: {file_size} bytes")

            with open(f"{store_name}.pkl", "rb") as f:
                try:
                    pickled_data = pickle.load(f)
                    st.write('Embeddings loaded from the disk')
                    embeddings = OpenAIEmbeddings()
                    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
                except EOFError:
                    st.write("Failed to load")
        else:
            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
            vectorstore._index = None
            pickled_data = {
                "custom_attributes": dir(vectorstore),
            }

            with open(f"{store_name}.pkl", "wb") as f:
                try:
                    pickle.dump(pickled_data, f)
                    st.write('Embeddings pickled successfully')
                except Exception as e:
                    st.write(f"Error during pickling: {e}")


        query = st.text_input("Ask questions:")

        if query:
            docs = vectorstore.similarity_search(query=query, k=3)

            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()
