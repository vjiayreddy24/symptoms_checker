import os
import re
import json
from typing import List, Optional, Union

import pandas as pd
from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

# ========== Load ENV ==========
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAOMp26ykP4NSselA6EpYQ1WEUZVEp-3Fc")
TEST_API_URL = os.getenv("TEST_API_URL", "http://localhost:9000/run_test")

# ======== File paths (EDIT THESE to your paths) =========
USER_DATA_CSV = r"C:\Users\vreddy_quantum-i\Desktop\Symptom_Checker\UserData.csv"
DOCTOR_DATA_CSV = r"C:\Users\vreddy_quantum-i\Desktop\Symptom_Checker\Data\doctor_data.csv"
FAISS_INDEX_PATH = r"C:\Users\vreddy_quantum-i\Desktop\Symptom_Checker\symptom_checker\faiss_index_dsm"
CHROMA_DB_PATH = "./chroma_db"  # directory for your disorder embeddings

# ========== App ==========
app = FastAPI(title="Unified Hospital AI API", version="1.0")

# ========== Imports that rely on packages ==========
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import chromadb

# Google Generative AI (Gemini) via LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
if not GEMINI_API_KEY:
    print("⚠️ GEMINI_API_KEY not set; symptom follow-ups & summarization will fail.")

# ========== Embeddings / Retriever ==========
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS retriever (if available)
try:
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print("✅ Retriever initialized successfully")
except Exception as e:
    retriever = None
    print("❌ Failed to initialize retriever:", e)

# LLM (Gemini) for follow-ups & summary
llm = None
try:
    if GEMINI_API_KEY:
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.0-flash-lite",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3,
        )
except Exception as e:
    print("❌ Failed to initialize LLM:", e)

# ========== Prompt ==========
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

# ========== Schemas ==========
class SymptomInput(BaseModel):
    symptom: str

class AnswerInput(BaseModel):
    symptom: str
    questions: List[str]
    answers: List[str]

class SummaryInput(BaseModel):
    summary: str

class AppointmentFromUserId(BaseModel):
    user_id: int
    date: str  # YYYY-MM-DD

class ManualBookingRequest(BaseModel):
    department: str
    disorder: str
    date: str  # YYYY-MM-DD

class RouteBookingRequest(BaseModel):
    user_id: int
    date: str  # YYYY-MM-DD

class BookChoice(BaseModel):
    choice: str  # "yes" or "no"

# ========== Utils ==========
def _weekday(date_str: str) -> str:
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        return date_obj.strftime("%a")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

def _extract_follow_up_questions(llm_output: str) -> List[str]:
    matches = re.findall(r"^\s*\d+\.\s*(.+)$", llm_output, flags=re.MULTILINE)
    return [q.strip() for q in matches if q.strip()]

def _generate_followups(symptom: str) -> List[str]:
    if retriever is None or llm is None:
        return []
    docs = retriever.get_relevant_documents(symptom)
    context_text = "\n\n".join(d.page_content for d in docs) if docs else ""
    prompt_text = FOLLOWUP_PROMPT_TEMPLATE.format(context=context_text, question=symptom)
    resp = llm.invoke(prompt_text)
    return _extract_follow_up_questions((resp.content or "").strip())

def _summarize(symptom: str, answers: List[str], questions: List[str]) -> str:
    if llm is None:
        return f"Primary symptom: {symptom}. Answers: {answers}. Questions: {questions}."
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

def _map_disorder_from_summary(summary_text: str) -> List[str]:
    # Chroma vector search for the best matching disorders
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name="disorders")
    query_vec = embedding_model.embed_query(summary_text)
    results = collection.query(query_embeddings=[query_vec], n_results=5)

    filtered = []
    threshold = 0.1
    if results and results.get("metadatas") and results["metadatas"][0]:
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            sim = 1.0 - float(dist)
            if sim >= threshold and "disorder" in meta:
                filtered.append(meta["disorder"])
        if filtered:
            return filtered
        # Fallback top-2
        return [m.get("disorder", "Unknown") for m in results["metadatas"][0][:2]]
    return []

# Mapping test codes → disorder names
DISORDER_MAPPING = {
    "AQ-10": "Autism Spectrum Disorder",
    "ASRS": "ADHD",
    "GAD-7": "Anxiety",
    "PHQ-9": "Depression",
    "Y-BOCS": "OCD",
    "PCL-5": "PTSD",
    "AUDIT": "Alcohol Use Disorder",
    "DAST": "Drug Use Disorder",
    "Mood Disorder Questionnaire": "Bipolar Disorder",
    "MSI-BPD": "BPD",
    "EAT-26": "Eating Disorders",
}

def _load_user_df() -> pd.DataFrame:
    df = pd.read_csv(USER_DATA_CSV)
    df.columns = df.columns.str.strip()
    return df

def _load_doctors_df() -> pd.DataFrame:
    df = pd.read_csv(DOCTOR_DATA_CSV)
    df.columns = df.columns.str.strip()
    return df

def _user_test_payload(user_id: int) -> Optional[dict]:
    """Return dict of tests for a given user_id, or None if missing."""
    df = _load_user_df()
    if user_id not in df["User ID"].values:
        return None
    user_record = df[df["User ID"] == user_id].iloc[0]
    if pd.isna(user_record.get("Test Taken")) or str(user_record.get("Test Taken")).strip() == "":
        return {}
    try:
        return json.loads(user_record["Test Taken"])
    except Exception:
        return {}

def _department_disorder_from_tests(test_results: dict) -> Optional[dict]:
    """Use your logic: severe → Psychiatry + that disorder; else Psychology."""
    if not test_results:
        return None
    # Prioritize "severe"
    for code, severity in test_results.items():
        if str(severity).strip().lower() == "severe" and code in DISORDER_MAPPING:
            return {"department": "Psychiatry", "disorder": DISORDER_MAPPING[code], "stage": "Severe"}
    # Otherwise collect mapped disorders and default Psychology
    mapped = [DISORDER_MAPPING[code] for code in test_results if code in DISORDER_MAPPING]
    if mapped:
        return {"department": "Psychology", "disorder": mapped[0], "stage": "Mild/Moderate"}
    return None

def _filter_doctors(department: str, disorder: str, date_str: str):
    weekday = _weekday(date_str)
    doctors_df = _load_doctors_df()
    filtered = doctors_df[
        (doctors_df["Specialization"].str.lower() == department.lower()) &
        (doctors_df["Disorder"].str.lower() == disorder.lower()) &
        (doctors_df["Available Days"].str.contains(weekday, na=False))
    ]
    return weekday, filtered

def _build_doctors_payload(filtered: pd.DataFrame, weekday: str):
    if filtered.empty:
        return []
    doctors = []
    for _, row in filtered.iterrows():
        slots = []
        try:
            slots = [s.strip() for s in str(row["Available Slots"]).split(",") if str(row["Available Slots"]).strip()]
        except Exception:
            slots = []
        doctors.append({
            "doctor_id": row.get("Doctor ID"),
            "doctor_name": row.get("Doctor Name"),
            "specialization": row.get("Specialization"),
            "disorder": row.get("Disorder"),
            "day": weekday,
            "available_slots": slots
        })
    return doctors

# ========== External Test API ==========
import requests

def call_test_api(user_id: int) -> dict:
    """POST {user_id} to TEST_API_URL. Expected response:
       { "disorder": "...", "stage": "Mild|Moderate|Severe", "department": "Psychiatry|Psychology" }
       If your Test API returns only scores, adapt here.
    """
    try:
        resp = requests.post(TEST_API_URL, json={"user_id": user_id}, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"Test API failed with status {resp.status_code}", "raw": resp.text}
    except Exception as e:
        return {"error": f"Test API exception: {e}"}

# ========== SYMPTOM CHECKER ENDPOINTS ==========
@app.post("/get_followups")
def get_followups(data: SymptomInput):
    questions = _generate_followups(data.symptom)
    # If retriever/LLM unavailable, you’ll get []
    return {"questions": questions}

@app.post("/summarize")
def summarize(data: AnswerInput):
    if len(data.questions) != len(data.answers):
        raise HTTPException(status_code=400, detail="questions and answers length mismatch")
    summary = _summarize(data.symptom, data.answers, data.questions)
    return {"summary": summary}

@app.post("/map_disorder")
def map_disorder_api(data: SummaryInput):
    disorders = _map_disorder_from_summary(data.summary)
    return {"summary": data.summary, "possible_disorders": disorders}

# ========== BOOKING CORE (your logic) ==========
@app.post("/book_and_find_doctor")
def book_and_find_doctor(req: AppointmentFromUserId):
    """Use UserData.csv → tests → derive department+disorder → filter doctors."""
    df = _load_user_df()
    if req.user_id not in df["User ID"].values:
        raise HTTPException(status_code=404, detail="User not found")

    test_payload = _user_test_payload(req.user_id)
    if test_payload is None:
        raise HTTPException(status_code=404, detail="User record not found")
    if test_payload == {}:
        return {"error": f"User {req.user_id} has no test records. Please take a test first."}

    deduced = _department_disorder_from_tests(test_payload)
    if not deduced:
        return {"error": "Unable to infer department/disorder from tests."}

    department = deduced["department"]
    disorder = deduced["disorder"]
    stage = deduced["stage"]

    weekday, filtered = _filter_doctors(department, disorder, req.date)
    doctors = _build_doctors_payload(filtered, weekday)
    if not doctors:
        return {"message": f"No doctors available in {department} for {disorder} on {weekday} ({req.date})."}

    return {
        "user_id": req.user_id,
        "department": department,
        "disorder": disorder,
        "stage": stage,
        "date": req.date,
        "day": weekday,
        "available_doctors": doctors
    }

# ========== DIRECT MANUAL BOOKING ==========
@app.post("/find_doctors")
def find_doctors(req: ManualBookingRequest):
    weekday, filtered = _filter_doctors(req.department, req.disorder, req.date)
    if filtered.empty:
        # Fallback: show top 3 by department if disorder day match failed
        doctors_df = _load_doctors_df()
        alt = doctors_df[
            (doctors_df["Specialization"].str.lower() == req.department.lower()) &
            (doctors_df["Available Days"].str.contains(weekday, na=False))
        ].head(3)
        if alt.empty:
            return {"message": f"No doctors available in {req.department} on {weekday} ({req.date})."}
        return {
            "department": req.department,
            "disorder": req.disorder,
            "date": req.date,
            "day": weekday,
            "available_doctors": _build_doctors_payload(alt, weekday)
        }

    return {
        "department": req.department,
        "disorder": req.disorder,
        "date": req.date,
        "day": weekday,
        "available_doctors": _build_doctors_payload(filtered, weekday)
    }

# ========== WRAPPED FLOWS ==========
@app.post("/flow/symptom_to_booking")
def flow_symptom_to_booking(req: AppointmentFromUserId):
    """
    Flow 1: (frontend) run symptom-checker → (backend) call Test API → stage → ask 'book?'
    For simplicity here we just run Test API + show doctors.
    """
    # 1) Try user's saved tests first
    test_payload = _user_test_payload(req.user_id)
    deduced = _department_disorder_from_tests(test_payload) if test_payload else None

    # 2) If no tests saved, call external Test API
    if not deduced:
        api_res = call_test_api(req.user_id)
        if "error" in api_res:
            return api_res
        # Expecting: disorder, stage, (optional) department; fallback rules:
        department = api_res.get("department")
        disorder = api_res.get("disorder")
        stage = api_res.get("stage", "Unknown")
        if not department:
            # Severe → Psychiatry else Psychology
            department = "Psychiatry" if stage.lower() == "severe" else "Psychology"
    else:
        department = deduced["department"]
        disorder = deduced["disorder"]
        stage = deduced["stage"]

    weekday, filtered = _filter_doctors(department, disorder, req.date)
    doctors = _build_doctors_payload(filtered, weekday)
    return {
        "user_id": req.user_id,
        "department": department,
        "disorder": disorder,
        "stage": stage,
        "date": req.date,
        "day": weekday,
        "available_doctors": doctors
    }

@app.post("/flow/direct/manual")
def flow_direct_manual(req: ManualBookingRequest):
    """Flow 2A: Direct → Book manually (dept+disorder+date)."""
    return find_doctors(req)

@app.post("/flow/direct/check_symptoms_and_book")
def flow_direct_check_symptoms_and_book(req: AppointmentFromUserId):
    """Flow 2B: Direct → Check symptoms & book (calls the same symptom→test→booking pipeline)."""
    return flow_symptom_to_booking(req)

@app.post("/flow/check_tests_and_book")
def flow_check_tests_and_book(req: RouteBookingRequest):
    """
    Check if the user has tests:
      - If not: call Test API, then show doctors
      - If yes: use those results to show doctors
    """
    return flow_symptom_to_booking(AppointmentFromUserId(user_id=req.user_id, date=req.date))

# ========== Helper endpoints ==========
@app.get("/user/{user_id}/tests")
def get_user_tests(user_id: int):
    payload = _user_test_payload(user_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="User not found")
    if payload == {}:
        return {"user_id": user_id, "tests": None, "message": "No tests found"}
    return {"user_id": user_id, "tests": payload}

@app.post("/should_book")
def should_book(choice: BookChoice):
    """
    Simple helper for frontends. If 'yes', they can proceed to a booking endpoint.
    """
    return {"proceed_to_booking": choice.choice.strip().lower() == "yes"}
