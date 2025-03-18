#!/usr/bin/env python3

from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from langchain_community.embeddings import Nomic
from langchain_community.embeddings import OllamaEmbeddings
from MistralEmbedder import embedder

#Charger les documents depuis un dossier 
def load_docs(folderDirectory) -> list:
    loader = PyPDFDirectoryLoader(folderDirectory)
    docs = loader.load()
    return docs

#Split les différents documents présents
def split(docs) -> list:
    texSpliter = RecursiveCharacterTextSplitter(
        chunk_size = 100 ,
        chunk_overlap = 20,
        length_function = len
    )
    return texSpliter.split_documents(docs)

#Initialiser la db
def initDb(chunks,folderPath,embeddings):
    faiss_path = os.path.join(folderPath,"index.faiss")
    if os.path.exists(faiss_path):
        vectorstore = FAISS.load_local(folder_path=folderPath, embeddings=embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(documents=chunks,embedding=embeddings)
        vectorstore.save_local(folderPath)
    return vectorstore

# def embedQuery(query ,embedding):
#     return embedding.embed_query(query)

def main():
    path = "./Docs"
    embedding= OllamaEmbeddings(
        model="nomic-embed-text")
    docs = load_docs(path)
    chunks = split(docs)
    vectoreStore = initDb(chunks=chunks, folderPath= path,embeddings=embedding)
    query = input ("Pose ta question : ")
    results = vectoreStore.similarity_search(query=query)    #Cela réenvoi une liste 
    l = []
    taille = 0
    for r in results:
         l.append(r.page_content)
    if len(l) < 2 :
        for r in l :
            print(r)
            taille += 1
    else:
        for i in range(2):
            print(l[i])
            print("\n\n")
            taille +=1
    print("Le nombre de chunk retourné est : ",taille)
    
if __name__=="__main__":
    main()