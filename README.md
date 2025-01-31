# Cocktail Advisor Chatbot

## Overview
The **Cocktail Advisor Chatbot** is a Python-based chat application that integrates a **large language model (LLM)** using **Retrieval-Augmented Generation (RAG)** with a **vector database**. The chatbot can:
- Answer cocktail-related knowledge base queries
- Recommend cocktails based on user preferences
- Provide instructions on how to make specific cocktails

The chatbot is trained on a dataset of cocktail recipes and uses **Streamlit** for the frontend and **FastAPI** for the backend.

## Features
### **Knowledge Base Queries**
The chatbot answers:
- `What are the 5 cocktails containing lemon?`
- `What are the 5 non-alcoholic cocktails containing sugar?`
- `What are my favorite ingredients?`

### **Advisor Queries**
The chatbot recommends cocktails based on:
- `Recommend 5 cocktails that contain my favorite ingredients.`
- `Recommend a cocktail similar to "Hot Creamy Bush".`

It retrieves and ranks relevant drinks using a **vector store** and **LangChain**.

## Installation
### **1. Clone the Repository**
```bash
git clone https://github.com/your-repo/cocktail-chatbot.git
cd cocktail-chatbot
```

### **2. Create and Activate Virtual Environment**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Download & Prepare Dataset**
```bash
python prepare_dataset.py
```

### **5. Fine-Tune the Model**
```bash
python fine_tune_with_val.py
```

### **6. Start the Backend API**
```bash
uvicorn main:app --reload
```

### **7. Start the Streamlit UI**
```bash
streamlit run streamlit_app.py
```

## API Usage
### **POST /chat**
Send a chat message to the bot.
```json
{
  "user_message": "How to make Mojito?"
}
```
**Response:**
```json
{
  "answer": "Muddle mint leaves, lime, and sugar. Add ice and rum. Stir well. Top with soda water.",
  "sources": ["Mojito"]
}
```

## Streamlit Frontend
The Streamlit interface allows users to interact with the chatbot. The frontend sends messages to the FastAPI backend and displays responses in a chat format.

## Training & Fine-Tuning
### If you don't want to use trained model - skip this step
### **Dataset Preparation**
We generate a dataset containing structured queries and responses:
- `"How to make Mojito?" → "Muddle mint leaves, lime, and sugar..."`
- `"What are the 5 cocktails containing lemon?" → "Margarita, Whiskey Sour..."`

```bash
python prepare_dataset.py
```

### **Fine-Tuning the Model**
```bash
python fine_tune_with_val.py
```
The model is trained on a dataset of cocktail recipes, ensuring it can correctly retrieve and generate relevant responses.

## RAG Implementation
### **1. Vector Store**
We use **FAISS** to store cocktail embeddings and retrieve relevant drinks based on similarity.

### **2. Retrieval Optimization**
- **Exact Match Retrieval:** Ensures correct cocktail instructions are provided.
- **Filtering by Ingredients:** Removes drinks with excluded ingredients.
- **Prompt Engineering:** Forces LLM to return accurate instructions.

## Troubleshooting
### **1. Model Not Responding Properly**
- Ensure dataset has been regenerated with:
  ```bash
  python prepare_dataset.py
  ```
- Ensure fine-tuning was successful with:
  ```bash
  python fine_tune_with_val.py
  ```

### **2. GPU Not Being Used**
- Install the correct **CUDA version**:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- Restart the training script.

## Future Improvements
- **Expand dataset** to include more cocktail variations.
- **Optimize prompt tuning** to improve generation quality.
- **Deploy as a web application** with frontend UI.

## Contributing
Pull requests are welcome! Open an issue for feature requests or bug reports.

## Conclusions

The Cocktail Advisor Chatbot successfully integrates FastAPI, Streamlit, and RAG to create an intelligent chatbot for cocktail recommendations and instructions. By fine-tuning a Flan-T5-based LLM, leveraging FAISS for vector retrieval, and optimizing prompts, the system delivers highly relevant and structured responses. Future improvements could include enhancing the dataset, refining prompt tuning, and introducing a full-fledged web-based UI for better user experience.

---

**Now your Cocktail Advisor Chatbot is ready for use!** 🍹

