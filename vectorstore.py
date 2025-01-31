import pandas as pd
from langchain.docstore.document import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def load_drinks_data(csv_path: str):
    """
    Loads drink data from a CSV file and converts it into a list of LangChain Document objects.
    
    Args:
        csv_path (str): Path to the CSV file containing drink data.
    
    Returns:
        list: A list of Document objects containing drink information.
    """
    df = pd.read_csv(csv_path)
    docs = []
    
    for _, row in df.iterrows():
        drink_id = row.get("id", "")
        name = row.get("name", "")
        alcoholic = row.get("alcoholic", "")
        category = row.get("category", "")
        glass_type = row.get("glassType", "")
        instructions = row.get("instructions", "")
        thumbnail = row.get("drinkThumbnail", "")
        ingredients = row.get("ingredients", "")
        measures = row.get("ingredientMeasures", "")

        content = (
            f"Drink ID: {drink_id}\n"
            f"Name: {name}\n"
            f"Alcoholic: {alcoholic}\n"
            f"Category: {category}\n"
            f"Glass Type: {glass_type}\n"
            f"Ingredients: {ingredients}\n"
            f"Measures: {measures}\n"
            f"Instructions: {instructions}\n"
            f"Thumbnail: {thumbnail}\n"
        )
        
        metadata = {
            "id": drink_id,
            "name": name,
            "alcoholic": alcoholic,
            "category": category,
            "thumbnail": thumbnail
        }
        
        docs.append(Document(page_content=content, metadata=metadata))
    
    return docs

def build_vectorstore(docs, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Builds a FAISS vector store from a list of Document objects using a Hugging Face embedding model.
    
    Args:
        docs (list): A list of Document objects.
        model_name (str): The name of the Hugging Face embedding model to use. Default is 'sentence-transformers/all-MiniLM-L6-v2'.
    
    Returns:
        FAISS: A FAISS vector store containing embedded drink documents.
    """
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = FAISS.from_documents(docs, embedder)
    return vectorstore

def search_top_k(vectorstore, query: str, k=5):
    """
    Searches the FAISS vector store for the top-k most similar drinks based on the given query.
    Filters out drinks containing excluded ingredients.
    """
    results = vectorstore.similarity_search(query, k=k)
    return results[:k]
