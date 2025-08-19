# === Helpers (your logic) =====================================================
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import re
import chromadb

app = FastAPI()

from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

from langchain_community.vectorstores import FAISS
try:
    vectorstore = FAISS.load_local(
        r"C:\Users\aamreen_quantum-i\OneDrive\Desktop\Symptoms_checker\symptoms_checker\faiss_index_dsm",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print("✅ Retriever initialized successfully")
except Exception as e:
    retriever = None
    print("❌ Failed to initialize retriever:", e)

# --- Load Local / API LLM ---
from langchain_google_genai import ChatGoogleGenerativeAI
gemini_api_key = "AIzaSyAOMp26ykP4NSselA6EpYQ1WEUZVEp-3Fc"
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash-lite",
    google_api_key=gemini_api_key,
    temperature=0.3,
)

# If you already have a prompt object, keep using it.
FOLLOWUP_PROMPT_TEMPLATE = """You are a medically cautious assistant helping to narrow down possible disorders.

You will receive:
- Context: Retrieved from a trusted knowledge base of disorders.
- Current patient information: What is already known from prior conversation.

Your task:
1. ONLY use the given context to decide which additional questions would most effectively narrow down the possible disorders.
2. The questions must be specific, relevant, and fact-based from the context.
3. Avoid general questions; focus on differentiating between similar disorders mentioned in the context.
4. Do NOT invent symptoms, facts, or questions not supported by the context.
5. If the context does not contain enough information to generate follow-up questions, respond with:
   "I could not find enough information in the provided context to generate follow-up questions."
6. Ask up to 3 follow-up questions only and make sure they're the best ones.

Format your output exactly as:
Follow-up Questions:
1. ...
2. ...
3. ...

Context:
{context}

Current Patient Information:
{question}
"""

# === Request Models ===
# === Request Models ===
class SymptomInput(BaseModel):
    symptom: str

class AnswerInput(BaseModel):
    symptom: str
    questions: List[str]
    answers: List[str]

class SummaryInput(BaseModel):
    summary: str


# === Helpers ===

def _extract_follow_up_questions(llm_output: str) -> List[str]:
    """Regex to extract numbered questions from LLM output."""
    matches = re.findall(r"^\s*\d+\.\s*(.+)$", llm_output, flags=re.MULTILINE)
    return [q.strip() for q in matches if q.strip()]

def _generate_followups(symptom: str) -> List[str]:
    docs = retriever.get_relevant_documents(symptom)
    context_text = "\n\n".join(d.page_content for d in docs)

    prompt_text = FOLLOWUP_PROMPT_TEMPLATE.format(
        context=context_text,
        question=symptom
    )
    resp = llm.invoke(prompt_text)
    return _extract_follow_up_questions(resp.content or "")

def _summarize(symptom: str, answers: List[str], questions: List[str]) -> str:
    qa_pairs = list(zip(questions, answers))
    formatted_qa = "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs])

    summarizer_prompt = f"""
You are a medical summarization assistant.

Task:
Summarize the patient's information into a **single, coherent paragraph** that clearly states:
- The primary symptom
- The answers to the follow-up questions

Rules:
1. Be clear and concise.
2. Do NOT add medical advice, opinions, or unrelated information.
3. Output only a narrative paragraph (no bullet points, no Q&A format).

Primary Symptom:
{symptom}

Follow-up Information:
{formatted_qa}
""".strip()
    resp = llm.invoke(summarizer_prompt)
    return (resp.content or "").strip()

def _map_disorder(summary_text: str) -> List[str]:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(name="disorders")

    query_vec = embedding_model.embed_query(summary_text)
    results = collection.query(query_embeddings=[query_vec], n_results=5)

    filtered = []
    threshold=0.1
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        sim = 1.0 - float(dist)
        if sim >= threshold:
            filtered.append(meta["disorder"])

    if filtered:
        return filtered
    return [m["disorder"] for m in results["metadatas"][0][:2]]

# === Endpoints ===

@app.post("/get_followups")
def get_followups(data: SymptomInput):
    """Generate up to 3 follow-up questions for a given symptom."""
    questions = _generate_followups(data.symptom)
    return {"questions": questions}

@app.post("/summarize")
def summarize(data: AnswerInput):
    """Summarize symptom + answers into a clean note."""
    # regenerate the same followups (to align with answers)
    summary = _summarize(data.symptom, data.answers, data.questions)
    return {"summary": summary}

@app.post("/map_disorder")
def map_disorder_api(data: SummaryInput):
    """Map the summarized text to possible disorders."""
    # regenerate followups → summarize → map
    disorders = _map_disorder(data.summary)
    return {"summary": data.summary, "possible_disorders": disorders}
