import os
os.environ['OPENAI_API_KEY'] = 'openai_key_api' # 따옴표 내부에는 본인의 open ai key를 입력
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAI

if __name__ == '__main__':
    print('hi')
    pdf_path = 'C:/Users/cws/vector/UW_240524.pdf'
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever())
    res = qa.invoke("Give me the gist of ReAct in 10 sentences translated to Korean") # ("")내부에는 자연어로 원하는 프롬프트를 추출하도록 한다.
    print(res)