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

DRINKS_CSV = "data/drinks.csv"
drink_docs = load_drinks_data(DRINKS_CSV)
vectorstore = build_vectorstore(drink_docs)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

FINE_TUNED_MODEL_PATH = "./fine-tuned-flan-t5-cocktails"
BASE_MODEL_NAME = "google/flan-t5-base"

# Check if the fine-tuned model directory exists and contains necessary files
if os.path.isdir(FINE_TUNED_MODEL_PATH) and os.path.exists(f"{FINE_TUNED_MODEL_PATH}/config.json"):
    MODEL_NAME = FINE_TUNED_MODEL_PATH
    print("✅ Using fine-tuned model.")
else:
    MODEL_NAME = BASE_MODEL_NAME
    print("⚠️ Fine-tuned model not found. Falling back to base model.")
    
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.3,
)

local_llm = HuggingFacePipeline(pipeline=hf_pipeline)
stuff_prompt = PromptTemplate(
    template="""
You are a helpful cocktail assistant.

Below is relevant context from a drinks database:
{context}

QUESTION:
{question}

Requirements:
1. Provide a concise answer (up to {max_cocktails} cocktails if the user wants a list).
2. Focus on cocktail names or short summaries; do NOT output lengthy instructions or repeated lines.
3. Skip any drink that contains excluded ingredients.
4. If you have duplicates, skip them.

Final Answer:
""",
    input_variables=["context", "question", "max_cocktails"],
)

stuff_llm_chain = LLMChain(llm=local_llm, prompt=stuff_prompt)

stuff_chain = StuffDocumentsChain(
    llm_chain=stuff_llm_chain,
    document_separator="\n\n",
    document_variable_name="context",
)

USER_FAVORITE_INGREDIENTS = []
USER_EXCLUDED_INGREDIENTS = []

def parse_positive_ingredients(msg: str):
    """
    Parse statements like:
    - "I like rum, vodka"
    - "My favorite ingredients are gin, lemon"
    Return list of new liked ingredients (lowercased).
    """
    msg_lower = msg.lower()

    if "my favorite ingredients are" in msg_lower or "my favourite ingredients are" in msg_lower:
        idx = msg_lower.find("are") + len("are")
        sub = msg_lower[idx:].strip(" .:")
    elif "i like" in msg_lower:
        idx = msg_lower.find("i like") + len("i like")
        sub = msg_lower[idx:].strip(" .:")
    else:
        return []

    splitted = re.split(r"\s+and\s+|,", sub)
    new_likes = [s.strip().lower() for s in splitted if s.strip()]
    for ing in new_likes:
        if ing in USER_EXCLUDED_INGREDIENTS:
            USER_EXCLUDED_INGREDIENTS.remove(ing)
    return new_likes

def parse_negative_ingredients(msg: str):
    """
    Parse statements like:
    - "I don't like rum, vodka"
    - "I hate gin"
    - "I'm allergic to sugar"
    Return list of disliked/allergic ingredients (lowercased).
    """
    msg_lower = msg.lower()

    negative_phrases = ["i don't like", "i do not like", "i hate", "i'm allergic to", "i am allergic to"]
    found_phrase = None
    for phrase in negative_phrases:
        if phrase in msg_lower:
            found_phrase = phrase
            break
    
    if not found_phrase:
        return []

    idx = msg_lower.find(found_phrase) + len(found_phrase)
    sub = msg_lower[idx:].strip(" .:")

    splitted = re.split(r"\s+and\s+|,", sub)
    new_dislikes = [s.strip().lower() for s in splitted if s.strip()]

    for ing in new_dislikes:
        if ing in USER_FAVORITE_INGREDIENTS:
            USER_FAVORITE_INGREDIENTS.remove(ing)
    return new_dislikes

def is_asking_for_favorites(msg: str):
    msg_lower = msg.lower()
    return (
       "what are my favorite ingredients" in msg_lower
       or "which ingredients do i like" in msg_lower
    )

def is_asking_for_excluded(msg: str):
    msg_lower = msg.lower()
    return (
        "which ingredients do i hate" in msg_lower
        or "what am i allergic to" in msg_lower
        or "which ingredients do i not like" in msg_lower
        or "which ingredients do i dislike" in msg_lower
    )
def parse_number_of_cocktails(msg: str) -> int:
    """
    Simple regex to find "recommend X cocktails" or "give me X cocktails"
    If none found, default to 5.
    """
    pattern = r"(?:recommend|give me)\s+(\d+)\s+cocktails"
    match = re.search(pattern, msg.lower())
    if match:
        return int(match.group(1))
    else:
        # If the user doesn't specify a number, default to 5
        return 5
class ChatRequest(BaseModel):
    user_message: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    user_msg = req.user_message.strip()

    if is_asking_for_favorites(user_msg):
        if USER_FAVORITE_INGREDIENTS:
            favs_joined = ", ".join(USER_FAVORITE_INGREDIENTS)
            return ChatResponse(
                answer=f"You like these ingredients: {favs_joined}",
                sources=[]
            )
        else:
            return ChatResponse(
                answer="You haven't told me your favorite ingredients yet.",
                sources=[]
            )

    if is_asking_for_excluded(user_msg):
        if USER_EXCLUDED_INGREDIENTS:
            ex_joined = ", ".join(USER_EXCLUDED_INGREDIENTS)
            return ChatResponse(
                answer=f"You dislike or are allergic to: {ex_joined}",
                sources=[]
            )
        else:
            return ChatResponse(
                answer="You haven't told me which ingredients you dislike yet.",
                sources=[]
            )

    new_likes = parse_positive_ingredients(user_msg)
    if new_likes:
        added = []
        for ing in new_likes:
            if ing not in USER_FAVORITE_INGREDIENTS:
                USER_FAVORITE_INGREDIENTS.append(ing)
                added.append(ing)
        if added:
            return ChatResponse(
                answer=f"Noted! Added these as your favorite ingredients: {', '.join(added)}",
                sources=[]
            )
        else:
            return ChatResponse(
                answer="Those ingredients were already in your favorites.",
                sources=[]
            )

    new_dislikes = parse_negative_ingredients(user_msg)
    if new_dislikes:
        added = []
        for ing in new_dislikes:
            if ing not in USER_EXCLUDED_INGREDIENTS:
                USER_EXCLUDED_INGREDIENTS.append(ing)
                added.append(ing)
        if added:
            return ChatResponse(
                answer=f"Noted! You don't like or are allergic to: {', '.join(added)}",
                sources=[]
            )
        else:
            return ChatResponse(
                answer="You already excluded those ingredients.",
                sources=[]
            )

    n_cocktails = parse_number_of_cocktails(user_msg)

    fav_text = (
        "User's favorite ingredients: " + ", ".join(USER_FAVORITE_INGREDIENTS)
        if USER_FAVORITE_INGREDIENTS
        else "No favorites yet."
    )
    ex_text = (
        "User excludes: " + ", ".join(USER_EXCLUDED_INGREDIENTS)
        if USER_EXCLUDED_INGREDIENTS
        else "No excluded ingredients."
    )
    combined_query = f"{req.user_message}\n\nAdditional info:\n- {fav_text}\n- {ex_text}"

    retrieval_result = retriever.get_relevant_documents(combined_query)
    def has_excluded_ingredient(doc_text):
        return any(exc in doc_text for exc in USER_EXCLUDED_INGREDIENTS)

    def contains_any_favorite(doc_text):
        if not USER_FAVORITE_INGREDIENTS:
            return True
        return any(fav in doc_text for fav in USER_FAVORITE_INGREDIENTS)

    final_docs = []
    for doc in retrieval_result:
        doc_text_lower = doc.page_content.lower()
        if has_excluded_ingredient(doc_text_lower):
            continue
        if not contains_any_favorite(doc_text_lower):
            continue
        final_docs.append(doc)

    if not final_docs:
        return ChatResponse(
            answer="No suitable cocktails found after applying your likes/dislikes.",
            sources=[]
        )

    final_docs = final_docs[:n_cocktails]

    chain_input = {
        "input_documents": final_docs,
        "question": combined_query,
        "max_cocktails": str(n_cocktails)
    }
    chain_output = stuff_chain.run(chain_input)

    seen_cocktails = set()
    source_names = []
    for d in final_docs:
        name = d.metadata.get("name", "Unknown Cocktail")
        if name not in seen_cocktails:
            source_names.append(name)
            seen_cocktails.add(name)

    return ChatResponse(
        answer=chain_output.strip(),
        sources=source_names
    )
