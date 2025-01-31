import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document

from vectorstore import load_drinks_data, build_vectorstore

app = FastAPI()

# drink data
DRINKS_CSV = "data/drinks.csv"
drink_docs = load_drinks_data(DRINKS_CSV)
vectorstore = build_vectorstore(drink_docs)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# language model
MODEL_NAME = "./fine-tuned-flan-t5-cocktails" if "./fine-tuned-flan-t5-cocktails" else "google/flan-t5-base" # if trained model is generated - better use it
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# text generation pipeline
hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.3,
)

local_llm = HuggingFacePipeline(pipeline=hf_pipeline)

# prompt template
stuff_prompt = PromptTemplate(
    template="""
You are a professional cocktail assistant.

Below is relevant context from a drinks database:
{context}

QUESTION:
{question}

Guidelines:
1. Recommend up to {max_cocktails} cocktails based on user preferences.
2. Prioritize cocktails containing favorite ingredients.
3. Do NOT recommend drinks that contain excluded ingredients.
4. Keep responses concise (just cocktail names unless asked otherwise).
5. Ensure a diverse set of recommendations (avoid duplicates).

Final Answer:
""",
    input_variables=["context", "question", "max_cocktails"],
)


# LLM chain and document processing chain
stuff_llm_chain = LLMChain(llm=local_llm, prompt=stuff_prompt)
stuff_chain = StuffDocumentsChain(
    llm_chain=stuff_llm_chain,
    document_separator="\n\n",
    document_variable_name="context",
)

# User ingredient preferences
USER_FAVORITE_INGREDIENTS = []
USER_EXCLUDED_INGREDIENTS = []

# request and response models
class ChatRequest(BaseModel):
    user_message: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """
    FastAPI endpoint to handle user queries and return cocktail recommendations.
    """
    user_msg = req.user_message.strip()

    # user preference queries
    if "favorite ingredients" in user_msg:
        response_text = ", ".join(USER_FAVORITE_INGREDIENTS) if USER_FAVORITE_INGREDIENTS else "You haven't specified any favorite ingredients yet."
        return ChatResponse(answer=response_text, sources=[])
    
    if "excluded ingredients" in user_msg:
        response_text = ", ".join(USER_EXCLUDED_INGREDIENTS) if USER_EXCLUDED_INGREDIENTS else "You haven't specified any excluded ingredients yet."
        return ChatResponse(answer=response_text, sources=[])

    # Retrieved relevant documents from vectorstore
    retrieval_result = retriever.get_relevant_documents(user_msg)
    final_docs = [doc for doc in retrieval_result if doc.metadata.get("name") not in USER_EXCLUDED_INGREDIENTS]

    if not final_docs:
        return ChatResponse(answer="No suitable cocktails found based on your preferences.", sources=[])

    # LLM chain with the retrieved documents
    chain_output = stuff_chain.run({"input_documents": final_docs, "question": user_msg, "max_cocktails": "5"})
    sources = [doc.metadata.get("name", "Unknown Cocktail") for doc in final_docs]

    return ChatResponse(answer=chain_output.strip(), sources=sources)
