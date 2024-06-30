from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

def create_vectordb(embedding_model = 'all-MiniLM-L6-v2.gguf2.f16.gguf', input_path='data/processed_docs', output_path='data/vectordb'):

    loader = DirectoryLoader(input_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=32)
    chunks = splitter.split_documents(documents)

    embeddings = GPT4AllEmbeddings(model_name=embedding_model, gpt4all_kwargs={'allow_download': 'True'})

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(output_path)

if __name__ == "__main__":
    create_vectordb()