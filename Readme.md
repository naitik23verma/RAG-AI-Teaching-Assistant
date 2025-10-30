# RAG-Based AI Teaching Assistant

A Retrieval-Augmented Generation (RAG) based AI assistant that learns directly from your own videos and answers questions contextually.  
This assistant uses Whisper for transcription, embedding models for semantic understanding, and a local LLM for intelligent responses.

---

## 1. Collect Your Videos
Move all your video files into the `videos` folder.  
These will act as your knowledge base for the AI assistant.

---

## 2. Convert Videos to MP3
Run the `video_to_mp3.py` file.  
This script will automatically convert all video files in the `videos` folder into `.mp3` format and store them in the `audios` folder.

---

## 3. Convert MP3 Files to JSON
Run the `mp3_to_json.py` file.  
This step will transcribe each `.mp3` file into text using the Whisper model and store the output as structured JSON files inside the `jsons` folder.

Each JSON file contains timestamped text chunks extracted from your audio, making the data ready for embedding.

---

## 4. Convert JSON Files to Vectors
Use the `read_chunks.py` file.  
This script performs the following steps:
- Reads all JSON transcript files from the `jsons` folder.
- Creates vector embeddings for each text chunk using your embedding model (e.g., `bge-m3`).
- Stores the embeddings and related metadata in a DataFrame.
- Saves the resulting DataFrame as a `.joblib` pickle file for faster retrieval during inference.

---

## 5. Prompt Generation and LLM Response
Open and modify the `process_incoming.py` file.  
This is where your AI assistant becomes interactive. The file:
- Loads stored embeddings from the pickle file.
- Accepts user queries and converts them into embeddings.
- Finds the most relevant text chunks using cosine similarity.
- Constructs a contextual prompt with those chunks.
- Sends the prompt to the local LLM (for example, `llama3.2` via Ollama).
- Returns a smart, context-aware response.

---

## Folder Structure

RAG_based_AI_Assistant/
│
├── videos/ # Store your input video files
├── audios/ # Contains audio (.mp3) files
├── jsons/ # Contains transcribed text files
├── embeddings/ # Stores generated embeddings (.joblib)
│
├── video_to_mp3.py # Converts videos to MP3
├── mp3_to_json.py # Converts audio to JSON using Whisper
├── read_chunks.py # Generates embeddings and saves them
├── process_incoming.py # Handles queries and generates LLM responses
└── README.md   



---

## Tech Stack
- **Python**
- **Whisper (Hugging Face Transformers)**
- **Ollama (for Local LLMs)**
- **bge-m3 / Sentence Transformers (for embeddings)**
- **NumPy, Pandas, Scikit-learn, Joblib**

---

## Overview
This AI assistant is designed to teach or explain concepts from your own videos.  
It listens, learns, and answers questions contextually — just like a human teaching assistant trained on your specific lectures or tutorials.


## Author
**Naitik Verma**  
B.Tech CSE, NIT Bhopal (MANIT)  
Web Developer | AI & ML Enthusiast | Robotics Club, NIT Bhopal