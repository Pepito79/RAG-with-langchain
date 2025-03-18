#!/usr/bin/env python3
import os
import json
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader


path = "./Docs"
loader = PyPDFDirectoryLoader(path)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100 , 
    chunk_overlap = 30,
    length_function = len
)
liste = splitter.split_documents(docs)
print(liste)
