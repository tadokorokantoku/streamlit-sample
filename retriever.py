import pickle
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

## 作成済みのDocumentデータをload
with open('data/faq_document.pkl', 'rb') as f:
    docs = pickle.load(f)

def generateRetrievalModel():
    transformer = Html2TextTransformer(ignore_links=False)
    transformed_docs = transformer.transform_documents(docs)

    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(transformed_docs, embeddings)

    retriever = db.as_retriever()
    return retriever