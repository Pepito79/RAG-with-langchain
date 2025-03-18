#!/usr/bin/env python3

from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
# from langchain_community.embeddings import Nomic
from langchain_community.embeddings import OllamaEmbeddings
from MistralEmbedder import embedder
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import ollama

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

def llm_response(query, context):
    # Construction du prompt en utilisant un formatage propre
    prompt = f"""
    Réponds à la question en utilisant uniquement le contexte suivant:
    {context}.
    
    ---------------------------------------------------
    Réponds à la question suivante : {query} en utilisant uniquement le contexte mentionné ci-dessus.
    """
    # Création du template du prompt et application des valeurs
    prompt_template = ChatPromptTemplate.from_template(prompt)
    final_prompt = prompt_template.format(context=context, question_query=query)
    return final_prompt
  
def afficher_menu():
    print("Menu:")
    print("0 - Quitter ")
    print("1 - Questionner votre document ")
          
def main():
    query =""
    while True:
        afficher_menu()
        choix = int(input("Veuillez choisir une option : "))
        if choix == 0:
            print("Sorite du programme")
            break
        elif choix==1:
            query = input("Pose ta question : ")
        else:
            print("Erreur : Vous devez entrer 0 ou 1.")
        
        
        path = "./Docs"    #À modifier si vous avez un chemin différent
        embedding = OllamaEmbeddings(model="nomic-embed-text")  # Spécification du modèle d'embedding
        docs = load_docs(path)  # Chargement des documents
        chunks = split(docs)  # Découpage des documents en morceaux
        vectorStore = initDb(chunks=chunks, folderPath=path, embeddings=embedding)  # Initialisation de la base FAISS
        # Recherche des résultats les plus pertinents
        results = vectorStore.similarity_search(query=query)
        
        # Préparation des morceaux pertinents pour la réponse
        context = ""
        for r in results[:2]:  # Limite aux 2 premiers résultats pour éviter d'être trop verbeux
            context += r.page_content + "\n\n"
            
        # # Affichage des résultats sélectionnés
        # print("Contexte utilisé pour répondre à la question:")
        # print(context)
            
        # Création du prompt pour le modèle LLM
        prompt = llm_response(query=query, context=context)
            
        # Utilisation du modèle LLM pour générer une réponse
        model = ollama.Ollama(model="mistral")  # Initialisation du modèle Ollama avec Mistral
        response = model.invoke(prompt)  # Appel au modèle pour obtenir une réponse
            
        # Affichage de la réponse générée par le modèle
        print("Réponse générée par le modèle :")
        print(response)
        
if __name__ == "__main__":
    main()
