#!/usr/bin/env python3
import requests as r
from langchain_community.embeddings import OllamaEmbeddings

def chat(query: str):
    """
    On donne une query et on effectue un POST request 
    et on prend le résultat.
    """
    
    # Envoi de la requête POST à l'API
    response = r.post(
        url="http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": query, "stream": False}
    )
    
    # Vérifie si la requête a échoué
    response.raise_for_status()
    
    # Récupère la réponse JSON
    result = response.json()
    
    # Retourne la réponse obtenue ou un message par défaut
    return result.get('response', 'Pas de réponse dans la requête')


def embedder(query):
    try:
        embedding_model = OllamaEmbeddings(
            model= "mistral",
            base_url="http://localhost:11434"
        )
        embedding  = embedding_model.embed_query(query)
        return embedding
    except Exception as e:
        print(f"Error connecting to Ollama API: {str(e)}")
        print("Please ensure Ollama is running with: 'sudo systemctl start ollama'")
        raise
    
if __name__ == "__main__":  
    
    msg = input("Text to embbed: ")  # Demande à l'utilisateur d'entrer une question
    embedd = embedder(msg)  # Appel à la fonction `chat` avec le message
    print("Réponse de l'API:", embedd)  # Affiche la réponse reçue
    print(len(embedd))
