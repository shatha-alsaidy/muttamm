from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
import os
import json
import sqlite3
import time
import uuid
import re
import base64
import logging
from io import BytesIO
from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
import whisper
from tiktoken import encoding_for_model
from datetime import datetime
import httpx

logger = logging.getLogger("muttamm_agent")
logger.setLevel(logging.INFO)
logging.getLogger("uvicorn").setLevel(logging.WARNING)

# ========== CONFIG ==========
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ========== Models & Clients ==========
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

persist_dir = "chroma_db_archive"

vectordb = Chroma(
    persist_directory=persist_dir, 
    embedding_function=embedding_model
)

# Whisper
'''try:
    whisper_model = whisper.load_model("large-v3")
except Exception as e:
    logger.warning("Failed to load whisper large-v3 locally; try 'large' or use OpenAI STT API. Error: %s", e)
    whisper_model = whisper.load_model("large")'''

DB_PATH = "absher_services_demo.db"

def uid() -> str:
    return str(uuid.uuid4())[:8]

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def make_response_item(rtype: str, message: str, options: Optional[List[str]] = None, input_key: Optional[str] = None,metadata: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
    return {"type": rtype, "message": message, "options": options or [], "input_key": input_key, "metadata": metadata or {}}

# ---------------- State schema ----------------
class MuttammState(TypedDict):
    user_id: str
    user_first_message: Optional[str]
    user_current_message: Optional[str]
    chat_history: List[Dict[str,str]]
    message_type: Optional[str]
    extracted_text: Optional[str]
    emotion: Optional[str]
    intent: Optional[str]
    service_code: Optional[str]
    service_step: Optional[str]
    collected_info: Dict[str, Any]
    collected: Dict[str, Any]
    pending_task: Optional[str]
    sql_result: Optional[Any]
    retrieved_context: Optional[List[str]]
    escalate: bool
    unsolved_count: int
    final_response: Dict[str, Any]
    voice_bytes: Optional[bytes]
    image_bytes: Optional[bytes]
    image_caption: Optional[str]
    reply_with_voice: bool
    response_item: Optional[Dict[str, Any]]

# -------------------------
# AbsherTools
# -------------------------
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()
# lookups
def lookup_service(code: str) -> Optional[Dict[str,Any]]:
    c.execute("SELECT code,name,category,description FROM services WHERE code=?", (code,))
    r = c.fetchone()
    if not r:
        return None
    return {"code": r[0], "name": r[1], "category": r[2], "description": r[3]}

def get_service_steps(service_code: str) -> List[Dict[str,Any]]:
    c.execute("SELECT step_id, step_order, title, description, ai_action FROM service_steps WHERE service_code=? ORDER BY step_order", (service_code,))
    rows = c.fetchall()
    return [{"step_id": r[0], "order": r[1], "title": r[2], "description": r[3], "ai_action": r[4]} for r in rows]

# user info
def get_user(user_id: int) -> Optional[Dict[str,Any]]:
    c.execute("SELECT id,name,national_id,phone,passport_expiry,id_expiry,license_expiry FROM users WHERE id=?", (user_id,))
    r = c.fetchone()
    if not r: return None
    return {"id": r[0], "name": r[1], "national_id": r[2], "phone": r[3], "passport_expiry": r[4], "id_expiry": r[5], "license_expiry": r[6]}

def normalize_image(image_bytes: bytes) -> Dict[str,Any]:
    return {"status": "normalized", "note": "image processed (demo stub)"}

def get_address(user_id:int) -> Dict[str,Any]:
    return {"address": "حي الروضة، شارع الملك فهد، الرياض", "can_update": True}

def get_fees_for_service(service_code: str) -> Dict[str,Any]:
    fees_map = {
        "renew_id": {"service_fee": 50, "delivery_fee": 20.0},
        "issue_passport": {"service_fee": 100, "delivery_fee": 0},
        "issue_license": {"service_fee": 50, "delivery_fee": 30.0},
    }
    return fees_map.get(service_code, {"service_fee":0.0, "delivery_fee":0.0})

def get_shipping_options() -> List[Dict[str,Any]]:
    return [{"method":"البريد السعودي","fee":25.0,"eta_days":5},{"method":"سمسا","fee":40.0,"eta_days":2}]

def pre_payment(user_id:int, amount:float) -> Dict[str,Any]:
    pay_id = uid()
    return {"payment_id": pay_id, "payment_url": f"https://pay.example/{pay_id}", "amount": amount}

def cre_order(user_id:int, service_code:str, amount:float, shipping_method:Optional[str]=None) -> Dict[str,Any]:
    order_id = uid()
    now = now_ts()
    c.execute("INSERT INTO orders (order_id, user_id, service_code, amount, shipping_method, status, created_at) VALUES (?,?,?,?,?,?,?)", (order_id, user_id, service_code, amount, shipping_method or "", "pending", now))
    conn.commit()
    return {"order_id": order_id, "status": "pending", "created_at": now}

def register_drivingSchool_appt(user_id:int, school:str, date_time:str) -> Dict[str,Any]:
    res_id = uid()
    return {"reservation_id": res_id, "school": school, "datetime": date_time, "status": "reserved"}

# simple service detector by keywords
def find_service_by_text(text:str) -> Optional[str]:
    t = (text or "").lower()
    if "هوية" in t or "تجديد الهوية" in t or "تجديد الهوية الوطنية" in t:
        return "renew_id"
    if "جواز" in t or "'اصدار جواز" in t or "جواز سفر" in t:
        return "issue_passport"
    if "رخصة" in t or "رخصة قيادة" in t:
        return "issue_license"
    return None

def verify_expiry(state: MuttammState) -> bool:
    user_id = state.get("user_id")
    user = get_user(user_id)
    if not user:
        return False
    expires_at = user.get("id_expiry")
      # Parse the expiry date
    try:
      if isinstance(expires_at, str):
          # handles 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'
          expires_dt = datetime.fromisoformat(expires_at)
      elif isinstance(expires_at, datetime):
          expires_dt = expires_at
      else:
          # fallback
          expires_dt = datetime.strptime(str(expires_at), "%Y-%m-%d")
    except Exception:
        # Can't parse date -> treat as not expiring soon
        return False

    # Check if expiry is within 180 days
    now = datetime.now(expires_dt.tzinfo) if expires_dt.tzinfo else datetime.now()
    days_left = (expires_dt - now).days
    return days_left <= 180
def ask_upload_photo(state: MuttammState) -> MuttammState:
    updates = {}
    service_info = ["التحقق من صلاحية الهوية", "فحص تاريخ انتهاء الهوية؛ إذا تبقى <=180 يوم يصبح مؤهلا"]
    response_item = make_response_item("upload", f"service_info: {service_info} \nالرجاء رفع الصورة الآن.",input_key="image")
    updates['service_step'] = state.get('service_step',0) + 3
    new_history = state.get("chat_history", []) + [{
            "role": "agent",
            "content": response_item
        }]
    updates["chat_history"] = new_history
    updates["pending_task"] = "await_user_input"
    updates["response_item"] = response_item
    return updates

def show_address(state: MuttammState) -> MuttammState:
    updates = {}
    service_info = ["عرض وتأكيد العنوان", "عرض عنوان المستخدم وطلب التأكيد أو التحديث"]
    addr = get_address(state.get("user_id"))
    response_item = make_response_item("options", f"service_info: {service_info}  {addr.get('address')}\nهذا عنوانك هل تريد تحديثه؟", input_key="confirm_address", options=["نعم","لا"])
    updates['service_step'] = state.get('service_step',0) + 1
    new_history = state.get("chat_history", []) + [{
            "role": "agent",
            "content": response_item
        }]
    updates["chat_history"] = new_history
    updates["pending_task"] = "await_user_input"
    updates["response_item"] = response_item
    return updates

def show_fees_and_confirm(state: MuttammState) -> MuttammState:
    updates = {}
    service_info = ["عرض الرسوم والحصول على الموافقة", "عرض رسوم الخدمة والتوصيل ومطالبة المستخدم بالموافقة"]
    service_code = state.get("service_code")
    fees = get_fees_for_service(service_code)
    response_item = make_response_item("confirm",f"service_info: {service_info} الرسوم: {fees.get('service_fee')} ريال + توصيل {fees.get('delivery_fee')} ريال.  هذه الرسوم الاجمالية للخدمة هل توافق عليها؟", input_key="confirm_fees", options=["نعم","لا"])
    updates['service_step'] = state.get('service_step',0) + 1
    new_history = state.get("chat_history", []) + [{
            "role": "agent",
            "content": response_item
        }]
    updates["chat_history"] = new_history
    updates["pending_task"] = "await_user_input"
    updates["response_item"] = response_item
    return updates

def present_payment(state: MuttammState) -> MuttammState:
    updates = {}
    user_id = state.get("user_id")
    service_code = state.get("service_code")
    collected = state.get("collected")
    fees = get_fees_for_service(service_code)
    shipping_fee = 0
    if collected.get("shipping_method"):
      opts = get_shipping_options()
      for opt in opts:
          if opt["method"] == collected.get("shipping_method"):
             shipping_fee = opt["fee"]

    amount = fees.get("service_fee",0) + fees.get("delivery_fee",0) + shipping_fee
    pay = pre_payment(user_id, amount)
    response_item = make_response_item("payment", f"اضغط لإتمام الدفع", input_key="payment_confirmed", metadata={"amount": amount})
    updates['service_step'] = state.get('service_step',0) + 1
    new_history = state.get("chat_history", []) + [{
            "role": "agent",
            "content": response_item
        }]
    updates["chat_history"] = new_history
    updates["pending_task"] = "await_payment"
    updates["response_item"] = response_item
    return updates

def create_order(state: MuttammState) -> MuttammState:
   updates = {}
   user_id = state.get("user_id")
   service_code = state.get("service_code")
   collected = state.get("collected")
   if not collected.get("payment_confirmed"):
          response_item = make_response_item("payment", "يجب إتمام الدفع قبل إنشاء الطلب.",input_key= "payment_confirmed")
          new_history = state.get("chat_history", []) + [{
                  "role": "agent",
                  "content": response_item
              }]
          updates["chat_history"] = new_history
          updates["pending_task"] = "await_payment"
          updates["response_item"] = response_item
          return updates
   fees = get_fees_for_service(service_code)
   shipping_fee = 0
   if collected.get("shipping_method"):
      opts = get_shipping_options()
      for opt in opts:
          if opt["method"] == collected.get("shipping_method"):
             shipping_fee = opt["fee"]
   amount = fees.get("service_fee",0) + fees.get("delivery_fee",0) + shipping_fee
   shipping = collected.get("shipping_method", "البريد السعودي")
   ord_res = cre_order(user_id, service_code, amount, shipping)
   response_item = make_response_item("done", f"تم إنشاء الطلب بنجاح. رقم الطلب: {ord_res.get('order_id')}", metadata={"amount": amount})
   updates['service_step'] = 1
   new_history = state.get("chat_history", []) + [{
            "role": "agent",
            "content": response_item
        }]
   updates["chat_history"] = new_history
   updates["pending_task"] = None
   updates["response_item"] = response_item
   updates['service_code'] = None
   return updates

def ask_passport_term(state: MuttammState) -> MuttammState:
    updates = {}
    service_info = ["اختيار مدة صلاحية الجواز", "يسأل المستخدم إن كان يفضل 5 أو 10 سنوات"]
    response_item = make_response_item("options", f"service_info: {service_info} اختر مدة صلاحية الجواز", options=["5 سنوات", "10 سنوات"],input_key= "passport_expiray")
    new_history = state.get("chat_history", []) + [{
            "role": "agent",
            "content": response_item
        }]
    updates['service_step'] = state.get('service_step',1) + 1
    updates["chat_history"] = new_history
    updates["pending_task"] = "await_user_input"
    updates["response_item"] = response_item
    return updates

def show_passport_details(state: MuttammState) -> MuttammState:
   updates = {}
   service_info = ["عرض بيانات الجواز والموافقة", "عرض تفاصيل الجواز وطلب موافقة المستخدم أو توجيهه للأحوال للتعديل"]
   user_id = state.get("user_id")
   user = get_user(user_id)
   passport_details = (
        f"الاسم الكامل بالعربية: {user.get('name')}\n"
        f"الاسم الكامل بالإنجليزية: {user.get('english_name')}\n"
        f"رقم الهوية الوطنية: {user.get('national_id')}\n"
        f"تاريخ الميلاد: {user.get('birthdate')}"
    )
   response_item = make_response_item("confirm", f"service_info: {service_info} {passport_details}, هل توافق على البيانات؟", input_key="confirm_details", options=["نعم","لا"])
   updates['service_step'] = state.get('service_step',0) + 1
   new_history = state.get("chat_history", []) + [{
            "role": "agent",
            "content": response_item
        }]
   updates["chat_history"] = new_history
   updates["pending_task"] = "await_user_input"
   updates["response_item"] = response_item
   return updates

def show_shipping_options(state: MuttammState) -> MuttammState:
   updates = {}
   service_info = ["عرض خيارات التوصيل والرسوم", "عرض خيارات البريد/سمسا والرسوم لكل خيار"]
   opts = get_shipping_options()
   response_item = make_response_item("options", f"service_info: {service_info} {opts}اختر وسيلة التوصيل", options=["البريد السعودي","سمسا"],input_key= "shipping_method")
   updates['service_step'] = state.get('service_step',0) + 1
   new_history = state.get("chat_history", []) + [{
            "role": "agent",
            "content": response_item
        }]
   updates["chat_history"] = new_history
   updates["pending_task"] = "await_user_input"
   updates["response_item"] = response_item
   return updates

def ask_region(state: MuttammState) -> MuttammState:
  updates = {}
  response_item = make_response_item("options", f"اختر منطقتك", options=["الرياض", "مكة المكرمة", "الشرقية"], input_key= "region")
  updates['service_step'] = state.get('service_step',1) + 1
  new_history = state.get("chat_history", []) + [{
          "role": "agent",
          "content": response_item
      }]
  updates["chat_history"] = new_history
  updates["pending_task"] = "await_user_input"
  updates["response_item"] = response_item
  return updates

def show_driving_schools(state: MuttammState) -> MuttammState:
  updates = {}
  schools = ["مدرسة جدة المتطورة", "مدارس الشميسي", "مدرسة الزهور"]
  response_item = make_response_item("options", "اختر مدرسة قيادة", options=schools, input_key= "school")
  updates['service_step'] = state.get('service_step',0) + 1
  new_history = state.get("chat_history", []) + [{
          "role": "agent",
          "content": response_item
      }]
  updates["chat_history"] = new_history
  updates["pending_task"] = "await_user_input"
  updates["response_item"] = response_item
  return updates

def pick_date_time(state: MuttammState) -> MuttammState:
  updates = {}
  response_item = make_response_item("options", "اختر التاريخ والوقت المناسب لموعدك", options=["2025-06-15 10:00","2025-06-20 14:00"],input_key="appt_date")
  updates['service_step'] = state.get('service_step',0) + 1
  new_history = state.get("chat_history", []) + [{
          "role": "agent",
          "content": response_item
      }]
  updates["chat_history"] = new_history
  updates["pending_task"] = "await_user_input"
  updates["response_item"] = response_item
  return updates

def confirm_registration(state: MuttammState) -> MuttammState:
  updates = {}
  user_id = state.get("user_id")
  collected = state.get("collected")
  school = collected.get("school", "مدرسة الزهور")
  dt = collected.get("appt_date", "2025-06-15 10:00")
  reg = register_drivingSchool_appt(user_id, school, dt)
  if not collected.get("confirm_res"):
      response_item = make_response_item("confirm", f"هل تؤكد التسجيل؟", input_key="confirm_res", options=["نعم","لا"])
      new_history = state.get("chat_history", []) + [{
              "role": "agent",
              "content": response_item
          }]
      updates["chat_history"] = new_history
      updates["pending_task"] = "await_user_input"
      updates["response_item"] = response_item
      return updates
  else:
      response_item = make_response_item("done", f"تم تأكيد التسجيل. رقم الحجز: {reg.get('reservation_id')}")
      updates['service_step'] = 1
      new_history = state.get("chat_history", []) + [{
              "role": "agent",
              "content": response_item
          }]
      updates["chat_history"] = new_history
      updates["pending_task"] = None
      updates["response_item"] = response_item
      updates['service_code'] = None
      return updates

def map_actions(state: MuttammState, ai_action: str) -> Any:
    if ai_action == "verify_expiry":
        return verify_expiry(state)
    elif ai_action == "ask_upload_photo":
        return ask_upload_photo(state)
    elif ai_action == "show_address":
        return show_address(state)
    elif ai_action == "show_fees_and_confirm":
        return show_fees_and_confirm(state)
    elif ai_action == "present_payment":
        return present_payment(state)
    elif ai_action == "create_order":
        return create_order(state)
    elif ai_action == "ask_passport_term":
        return ask_passport_term(state)
    elif ai_action == "show_passport_details":
        return show_passport_details(state)
    elif ai_action == "show_shipping_options":
        return show_shipping_options(state)
    elif ai_action == "ask_region":
        return ask_region(state)
    elif ai_action == "show_driving_schools":
        return show_driving_schools(state)
    elif ai_action == "pick_date_time":
        return pick_date_time(state)
    elif ai_action == "confirm_registration":
        return confirm_registration(state)
    else:
        return "not found"
    
def set_input(state: MuttammState) -> MuttammState:
    updates = {}
    EXPECTED_KEYS = ["confirm_address", "confirm_fees", "shipping_method", "confirm_res",
                     "passport_expiray", "confirm_details","region","school", "appt_date"]

    resp = state.get("response_item")
    if not resp:
        return updates

    input_key = resp.get("input_key")
    if not input_key or input_key not in EXPECTED_KEYS:
        return updates

    # ensure we don't mutate original state in-place
    collected = dict(state.get("collected", {}))
    user_value = state.get("extracted_text", "").strip()
    if user_value == "":
        # nothing to record
        return updates

    collected[input_key] = user_value
    collected['image'] = True
    collected["payment_confirmed"] = True
    updates["collected"] = collected
    # optionally clear the response_item or mark it answered
    updates["response_item"] = None
    return updates

def classify_service_intent_llm(user_message: str, service_context: str = "") -> str:
      prompt = f"""
      You are an assistant that classifies a user's message for a government service chatbot.

      The user message may fall into one of three categories:
      1. execute -> the user wants to perform a service (like issuing a passport, renewing an ID, etc.).
      2. step_response -> the user is providing input related to a specific step in an ongoing service flow (like selecting an appointment date, entering a number, etc.)
      or something showing an image or a file being uploaded or a payment being confirmed.
      3. info -> the user is asking general information about a service or procedures.

      User message: {user_message}
      Context: {service_context}

      Classify the user message into exactly one of these three categories and respond with only the category name.
      """
      try:
          response = openai_client.chat.completions.create(
              model="gpt-4o",
              messages=[{"role": "user", "content": prompt}],
              temperature=0,
              max_tokens=10
          )
          classification = response.choices[0].message.content.strip().lower()
          if classification not in ["execute", "step_response", "info"]:
              # fallback
              classification = "info"
          return classification
      except Exception as e:
          logger.warning("Service intent classification failed: %s", e)
          return "info"
      
# ========== Knowledge Retriever ==========
def knowledge_retriever(query: str, k: int = 10) -> List[str]:
    global vectordb
    if vectordb is None:
        return []
    try:
        results = vectordb.similarity_search(query, k=k)
        return [r.page_content for r in results] if results else []
    except Exception as e:
        logger.warning("Chroma retrieval error: %s", e)
        return []

# ========== Vision analysis helper  ==========
def gpt4o_image_analyze(image_bytes: bytes, caption: str = "") -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64}"
    prompt_text = (
        "You are an assistant that analyzes user-submitted images related to Absher services (e.g., ID cards, passports, residency permits, official documents, supporting files, or photos relevant to a service request)."
            "Your duties:"
            "1. Provide a clear general description of the image or document."
            "2. Identify whether the document appears complete and readable (no missing sections, no major obstructions, no severe blur)."
            "3. Extract any visible textual fields relevant to Absher workflows (e.g., ID number, passport number, name, expiry dates, residency number, reference numbers, etc.)."
            "4. Detect any potential issues unrelated to technical specifications (e.g., cropped information, illegible text, obstructed fields, mismatched document type)."
            "5. Assess whether the document seems appropriate for the intended service."
            "6. Output:"
              "- A concise English summary."
              "- A structured JSON object with the following keys:"
                "- summary"
                "- issues_found"
                "- issues_details"
                "- extracted_fields"
                "- approval_status (ready / needs_fix / unusable)"
                "- confidence (0–1)"
            f"Caption: {caption}"
    )
    resp = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You analyze images and produce structured outputs."},
            {"role": "user", "content": [
                {"type":"text","text": prompt_text},
                {"type":"image_url","image_url": {"url": data_uri}}
            ]}
        ],
        temperature=0.0,
    )
    txt = resp.choices[0].message.content.strip()
    return txt

# ========== Speech (Whisper) ==========
def transcribe_audio_bytes(audio_bytes: bytes) -> str:
    '''
    tmp_path = "/content/temp_audio.ogg"
    with open(tmp_path, "wb") as f:
        f.write(audio_bytes)
    try:
        res = whisper_model.transcribe(tmp_path, language="ar")
        return res.get("text","").strip()
    except Exception as e:
        logger.warning("Whisper local transcribe failed: %s", e)
        # fallback try without forcing language
        res = whisper_model.transcribe(tmp_path)
        return res.get("text","").strip()'''

# ========== ElevenLabs TTS ==========
async def tts_elevenlabs_arabic(text: str) -> bytes:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    payload = {"text": text, "voice_settings": {"stability":0.6,"similarity_boost":0.6}}
    async with httpx.AsyncClient() as client:
        r = await client.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        return r.content

# ========== Prompts ==========
EMOTION_PROMPT = """You are a short Arabic emotion classifier specialized on Saudi dialect.
Given this user message (Arabic), answer with one word: angry, frustrated, neutral, satisfied, grateful.
answer angry only if the user is very very angry, given an anger score of 4/5 or 5/5, and wait a bit for how the conversation turns out before answering angry.
Only return the word."""

INTENT_PROMPT = """You are a high-accuracy classifier for Absher government services.
Your task:
Given a Saudi-Arabic message, classify it into exactly ONE Absher service category:

1. "الأحوال المدنية"
   - Includes: الهوية الوطنية، تجديد الهوية، إصدار بدل فاقد، بياناتي، العنوان الوطني، سجل الأسرة

2. "الجوازات"
   - Includes: الجواز السعودي، إصدار جواز، تجديد جواز، السفر، الخروج والعودة، التأشيرات (للأفراد السعوديين)

3. "المرور"
   - Includes: رخصة القيادة، تجديد الرخصة، المخالفات، الاستعلام عن المركبات، تسجيل المركبات، الفحص الدوري

4. "شؤون الوافدين"
   - Includes: الإقامة، تجديد الإقامة، نقل الكفالة، التأشيرات المتعلقة بالوافدين، التابعين، الخروج والعودة للوافدين
Rules:
- Always pick **one** category.
- If the request is implicit, infer the closest category.
- If the message contains mixed content, pick the category MOST LIKELY being requested.
- Do NOT return anything except the category name.

Examples:
User: "ابي اجدد هويتي الوطنية"
→ "الأحوال المدنية"

User: "أصدّر جواز جديد لولدي"
→ "الجوازات"

User: "كم باقي على انتهاء الرخصة؟"
→ "المرور"

User: "ابي انقل العامل على مؤسستي"
→ "شؤون الوافدين"

User: "كيف اطلع سجل الاسرة؟"
→ "الأحوال المدنية"

User: "ابي أسدد مخالفة مرورية"
→ "المرور"

User: "كم رسوم تجديد إقامة العامل؟"
→ "شؤون الوافدين"

Return ONLY one of:
الأحوال المدنية
الجوازات
المرور
شؤون الوافدين
"""

RESPONSE_PROMPT_TEMPLATE = """
You are Muttamm, a smart assistant that helps users navigate Saudi government services covering four main categories
(الأحوال المدنية، الجوازات، المرور، شؤون الوافدين).

IMPORTANT:
- If there is an active response item in the chat history (meaning the agent is in the middle of a service step),
  you MUST rewrite the message inside that response item in natural Saudi Arabic.
- Do NOT ignore the response item.
- ONLY Consider the latest reponse item's message in chat history, use the previous ones to understand the context better.
- if the reponse item's message contains information the user needs to agree too, show all the information give (name, id, birthdate..ect).
- Rewrite the message using the context, user message, and the current service situation and make sure it is clear what is needed of the user.

General Guidelines:
- Always respond in clear and friendly Saudi dialect.
- If we are in a service workflow, continue it smoothly.
- If the user provided information requested earlier, integrate it and move to the next step.
- If the user is angry or frustrated, start with brief empathy, then guide them calmly.
- If there is retrieved context, use it naturally in your explanation.
- If an image was analyzed, acknowledge the findings only when relevant.
- Only escalate if the system indicates escalation is required.
- Keep responses short and practical (2–4 sentences).

Context:
{context}

User message:
{user_message}

Emotion: {emotion}
Detected Category: {intent}
latest response item: {response_item}

If we are in the middle of a service step and a response item exists:
→ Rewrite ONLY the message of that response item.

Otherwise:
→ Generate a natural Saudi response that fits the context and user need.

Produce the final answer in Saudi Arabic.
"""

# ---------------- LangGraph nodes ----------------
def detect_input_type_node(state: MuttammState) -> MuttammState:
    updates = {}
    input_updates = set_input(state)
    updates.update(input_updates)

    updates["user_id"] = 1

    mtype = state.get("message_type", "text")

    # ALWAYS reset voice reply flag
    updates["reply_with_voice"] = (mtype == "voice")
    if mtype == "text":
        updates["extracted_text"] = state.get("user_current_message","")
    return updates

def speech_to_text_node(state: MuttammState) -> MuttammState:
    updates = {}
    if state.get("message_type") == "voice" and state.get("voice_bytes"):
        updates["reply_with_voice"] = True
        text = transcribe_audio_bytes(state["voice_bytes"])
        updates["extracted_text"] = text
        new_history = state["chat_history"] + [{"role":"agent","content": text}]
        updates["chat_history"] = new_history
        updates["message_type"] = "text"
    return updates

def vision_analyzer_node(state: MuttammState) -> MuttammState:
    updates = {}
    if state.get("message_type") == "image" and state.get("image_bytes"):
        try:
            desc = gpt4o_image_analyze(state["image_bytes"], caption=state.get("image_caption",""))
            updates["extracted_text"] = desc
            new_history = state["chat_history"] + [{"role":"agent","content": f"[image_analysis]{desc}"}]
            updates["chat_history"] = new_history
            updates["message_type"] = "text"
        except Exception as e:
            logger.warning("vision analyze error: %s", e)
    return updates

def emotion_detection_node(state: MuttammState) -> MuttammState:
    updates = {}
    txt = state.get("extracted_text","")
    if not txt:
        return updates
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":"You are a concise classifier."},
                      {"role":"user","content": EMOTION_PROMPT + "\n\nMessage:\n" + txt}],
            temperature=0.0
        )
        out = resp.choices[0].message.content.strip().lower()
    except Exception as e:
        logger.warning("Emotion LLM error: %s", e)
        out = ""
    if out not in ["angry","frustrated","neutral","satisfied","grateful"]:
        if any(k in txt for k in ["غضب","غاضب","مغتاظ","مستاء"]):
            out = "angry"
        else:
            out = "neutral"
    updates["emotion"] = out
    return updates

def pre_escalation_check_node(state: MuttammState) -> MuttammState:
    updates = {}
    if state.get("emotion") == "angry":
        updates["escalate"] = True
    return updates

def intent_classification_node(state: MuttammState) -> MuttammState:
    updates = {}
    txt = state.get("extracted_text","")
    if not txt:
        return updates
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"system","content":"You are a short intent classifier."},
                      {"role":"user","content": INTENT_PROMPT + "\n\nMessage:\n" + txt}],
            temperature=0.0
        )
        intent = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Intent LLM error: %s", e)
        intent = ""
    possible = ["الأحوال المدنية", "الجوازات", "المرور", "شؤون الوافدين"]

    if intent not in possible:
        t = txt.lower()

        # الأحوال المدنية – Civil Affairs
        if any(w in t for w in [
            "هوية", "بطاقة", "سجل", "أسرة", "بدل فاقد", "بدل تالف",
            "مولود", "وفاة", "تعديل مهنة", "تحديث", "الأحوال"
        ]):
            intent = "الأحوال المدنية"

        # الجوازات – Passports
        elif any(w in t for w in [
            "جواز", "جوازات", "تجديد جواز", "خروج وعودة", "تمديد",
            "هوية مقيم", "الإقامة", "نقل خدمات", "تأشيرة"
        ]):
            intent = "الجوازات"

        # المرور – Traffic
        elif any(w in t for w in [
            "رخصة", "قيادة", "استمارة", "تفويض", "حادث", "مخالفة",
            "المرور", "نقل ملكية", "تجديد استمارة"
        ]):
            intent = "المرور"

        # شؤون الوافدين – Expatriate Affairs
        elif any(w in t for w in [
            "وافد", "كفيل", "نقل كفالة", "بلاغ هروب", "تأشيرة عمل",
            "تعديل مهنة عامل", "إلغاء هروب"
        ]):
            intent = "شؤون الوافدين"

        # Default fallback
        else:
            intent = "الأحوال المدنية"
    updates["intent"] = intent
    return updates

def router_node(state: MuttammState) -> MuttammState:
    updates = {}
    if state.get("escalate"):
        updates["pending_task"] = "escalate"
        return updates
    updates["pending_task"] = state.get("intent","الأحوال المدنية")
    return updates

# Sub-agent nodes
def ahwal_subagent_node(state: MuttammState) -> MuttammState:
    print("reached ahwal")
    updates = {}
    text = state.get("extracted_text", "")
    context = state.get("chat_history", [])
    service_type = classify_service_intent_llm(text, context)
    service_code = ""
    if service_type != "info" and state.get("service_code") is None:
        print(f"reached here: {service_type}")
        service_code = find_service_by_text(text)
        updates["service_code"] = service_code

    if service_type != "info":
      service_code = state.get("service_code") or service_code
      service_step = state.get("service_step", 1)
      print(service_code)
      steps = get_service_steps(service_code)
      print(steps)
      if not steps:
          print("entered wrong condition")
          response_item = make_response_item("text", "عذرًا، هذه الخدمة غير متاحة حاليا.")
          new_history = state.get("chat_history", []) + [{
                  "role": "agent",
                  "content": response_item
              }]
          updates["chat_history"] = new_history
          updates["response_item"] = response_item
          return updates

      # if finished
      if service_step > len(steps):
          response_item = make_response_item("done", f"انتهت إجراءات خدمة {service_code}.")
          new_history = state.get("chat_history", []) + [{
                  "role": "agent",
                  "content": response_item
              }]
          updates["chat_history"] = new_history
          updates["pending_task"] = None
          updates["response_item"] = response_item
          return updates

      step = steps[service_step - 1]
      ai_action = step["ai_action"]
      res = map_actions(state, ai_action)
      if res != "not found":
        if ai_action == "verify_expiry":
          if not res:
            response_item = make_response_item("done", f"لست مؤهل لتجديد الهوية إذ انها لم تقارب على الانتهاء")
            new_history = state.get("chat_history", []) + [{
                    "role": "agent",
                    "content": response_item
                }]
            updates["chat_history"] = new_history
            updates["pending_task"] = None
            updates["response_item"] = response_item
            return updates
          else:
            print("reached here, id is expired")
            step = steps[1]
            ai_action = step["ai_action"]
            print(ai_action)
            res = map_actions(state, ai_action)
            print(res)
            print("response_item" in res)

        print(step)
        print(res)

        if "response_item" in res:
            updates["response_item"] = res["response_item"]
        if "chat_history" in res:
            updates["chat_history"] = res["chat_history"]
        if "pending_task" in res:
            updates["pending_task"] = res["pending_task"]
        if "service_step" in res:
            updates["service_step"] = res["service_step"]
        if "service_code" in res:
            updates["service_code"] = res["service_code"]
        return updates
      else:
        response_item = make_response_item("done", f"المعذرة حصل خطأ ما")
        new_history = state.get("chat_history", []) + [{
                "role": "agent",
                "content": response_item
            }]
        updates["chat_history"] = new_history
        updates["pending_task"] = None
        updates["response_item"] = response_item
        return updates
    else:
        kr = knowledge_retriever(text, k=10)
        if not kr:
            updates["chat_history"] = state.get("chat_history", []) + [{
                "role": "agent",
                "content": "تقدر توضّح لي سؤالك أكثر؟"
            }]
            updates["pending_task"] = "await_clarification"
            return updates

        updates["retrieved_context"] = kr
        return updates


def jawazat_subagent_node(state: MuttammState) -> MuttammState:
    print("reached jawazat")
    updates = {}
    text = state.get("extracted_text", "")
    context = state.get("chat_history", [])
    service_type = classify_service_intent_llm(text, context)
    service_code = ""
    if service_type != "info" and state.get("service_code") is None:
        print(f"reached here: {service_type}")
        service_code = find_service_by_text(text)
        updates["service_code"] = service_code

    if service_type != "info":
      service_code = state.get("service_code") or service_code
      service_step = state.get("service_step", 1)
      print(service_code)
      steps = get_service_steps(service_code)
      print(steps)
      if not steps:
          print("entered wrong condition")
          response_item = make_response_item("text", "عذرًا، هذه الخدمة غير متاحة حاليا.")
          new_history = state.get("chat_history", []) + [{
                  "role": "agent",
                  "content": response_item
              }]
          updates["chat_history"] = new_history
          updates["response_item"] = response_item
          return updates

      # if finished
      if service_step > len(steps):
          response_item = make_response_item("done", f"انتهت إجراءات خدمة {service_code}.")
          new_history = state.get("chat_history", []) + [{
                  "role": "agent",
                  "content": response_item
              }]
          updates["chat_history"] = new_history
          updates["pending_task"] = None
          updates["response_item"] = response_item
          return updates

      step = steps[service_step - 1]
      ai_action = step["ai_action"]
      res = map_actions(state, ai_action)
      if res != "not found":
        if ai_action == "verify_expiry":
          if not res:
            response_item = make_response_item("done", f"لست مؤهل لتجديد الهوية إذ انها لم تقارب على الانتهاء")
            new_history = state.get("chat_history", []) + [{
                    "role": "agent",
                    "content": response_item
                }]
            updates["chat_history"] = new_history
            updates["pending_task"] = None
            updates["response_item"] = response_item
            return updates
          else:
            print("reached here, id is expired")
            step = steps[1]
            ai_action = step["ai_action"]
            print(ai_action)
            res = map_actions(state, ai_action)
            print(res)
            print("response_item" in res)

        print(step)
        print(res)

        if "response_item" in res:
            updates["response_item"] = res["response_item"]
        if "chat_history" in res:
            updates["chat_history"] = res["chat_history"]
        if "pending_task" in res:
            updates["pending_task"] = res["pending_task"]
        if "service_step" in res:
            updates["service_step"] = res["service_step"]
        if "service_code" in res:
            updates["service_code"] = res["service_code"]
        return updates
      else:
        response_item = make_response_item("done", f"المعذرة حصل خطأ ما")
        new_history = state.get("chat_history", []) + [{
                "role": "agent",
                "content": response_item
            }]
        updates["chat_history"] = new_history
        updates["pending_task"] = None
        updates["response_item"] = response_item
        return updates
    else:
        kr = knowledge_retriever(text, k=10)
        if not kr:
            updates["chat_history"] = state.get("chat_history", []) + [{
                "role": "agent",
                "content": "تقدر توضّح لي سؤالك أكثر؟"
            }]
            updates["pending_task"] = "await_clarification"
            return updates

        updates["retrieved_context"] = kr
        return updates


def murur_subagent_node(state: MuttammState) -> MuttammState:
    updates = {}
    text = state.get("extracted_text", "")
    context = state.get("chat_history", [])
    service_code = ""
    service_type = classify_service_intent_llm(text, context)
    if service_type != "info" and state.get("service_code") is None:
        service_code = find_service_by_text(text)
        updates["service_code"] = service_code

    if service_type != "info":
      service_code = state.get("service_code") or service_code
      service_step = state.get("service_step", 1)
      print(service_code)
      steps = get_service_steps(service_code)
      print(steps)
      if not steps:
          print("entered wrong condition")
          response_item = make_response_item("text", "عذرًا، هذه الخدمة غير متاحة حاليا.")
          new_history = state.get("chat_history", []) + [{
                  "role": "agent",
                  "content": response_item
              }]
          updates["chat_history"] = new_history
          updates["response_item"] = response_item
          return updates

      # if finished
      if service_step > len(steps):
          response_item = make_response_item("done", f"انتهت إجراءات خدمة {service_code}.")
          new_history = state.get("chat_history", []) + [{
                  "role": "agent",
                  "content": response_item
              }]
          updates["chat_history"] = new_history
          updates["pending_task"] = None
          updates["response_item"] = response_item
          return updates

      step = steps[service_step - 1]
      ai_action = step["ai_action"]
      res = map_actions(state, ai_action)
      if res != "not found":
        if ai_action == "verify_expiry":
          if not res:
            response_item = make_response_item("done", f"لست مؤهل لتجديد الهوية إذ انها لم تقارب على الانتهاء")
            new_history = state.get("chat_history", []) + [{
                    "role": "agent",
                    "content": response_item
                }]
            updates["chat_history"] = new_history
            updates["pending_task"] = None
            updates["response_item"] = response_item
            return updates
          else:
            print("reached here, id is expired")
            step = steps[1]
            ai_action = step["ai_action"]
            print(ai_action)
            res = map_actions(state, ai_action)
            print(res)
            print("response_item" in res)

        print(step)
        print(res)

        if "response_item" in res:
            updates["response_item"] = res["response_item"]
        if "chat_history" in res:
            updates["chat_history"] = res["chat_history"]
        if "pending_task" in res:
            updates["pending_task"] = res["pending_task"]
        if "service_step" in res:
            updates["service_step"] = res["service_step"]
        if "service_code" in res:
            updates["service_code"] = res["service_code"]
        return updates
      else:
        response_item = make_response_item("done", f"المعذرة حصل خطأ ما")
        new_history = state.get("chat_history", []) + [{
                "role": "agent",
                "content": response_item
            }]
        updates["chat_history"] = new_history
        updates["pending_task"] = None
        updates["response_item"] = response_item
        return updates
    else:
        kr = knowledge_retriever(text, k=10)
        if not kr:
            updates["chat_history"] = state.get("chat_history", []) + [{
                "role": "agent",
                "content": "تقدر توضّح لي سؤالك أكثر؟"
            }]
            updates["pending_task"] = "await_clarification"
            return updates

        updates["retrieved_context"] = kr
        return updates

def wafeedin_subagent_node(state: MuttammState) -> MuttammState:
    updates = {}
    text = state.get("extracted_text", "")
    context = state.get("chat_history", [])
    service_type = classify_service_intent_llm(text, context)

    if service_type != "info":
        response_item = make_response_item("text", "عذرًا، هذه الخدمة غير متاحة حاليا.")
        new_history = state.get("chat_history", []) + [{
                "role": "agent",
                "content": response_item
            }]
        updates["chat_history"] = new_history
        updates["response_item"] = response_item
        return updates

    else:
        kr = knowledge_retriever(text, k=10)
        if not kr:
            updates["chat_history"] = state.get("chat_history", []) + [{
                "role": "agent",
                "content": "تقدر توضّح لي سؤالك أكثر؟"
            }]
            updates["pending_task"] = "await_clarification"
            return updates

        updates["retrieved_context"] = kr
        return updates


def other_subagent_node(state: MuttammState) -> MuttammState:
    updates = {}
    text = state.get("extracted_text", "")

    kr = knowledge_retriever(text, k=10)
    if not kr:
        updates["chat_history"] = state.get("chat_history", []) + [{
            "role": "agent",
            "content": "تقدر توضّح لي سؤالك أكثر؟"
        }]
        updates["pending_task"] = "await_clarification"
        return updates

    updates["retrieved_context"] = kr
    return updates

def post_handler_node(state: MuttammState) -> MuttammState:
    updates = {}
    # Track unsolved queries
    if not state.get("response_item"):
        updates["unsolved_count"] = state.get("unsolved_count", 0) + 1
        if state.get("unsolved_count", 0) >= 30:
            updates["escalate"] = True
    return updates

def response_generator_node(state: MuttammState) -> MuttammState:
    updates = {}
    r = state.get("response_item",{})
    if state.get("escalate"):
        final_response = make_response_item("text","حسنًا، سأحوّل محادثتك الآن إلى موظف خدمة العملاء لمتابعة المشكلة. يرجى الانتظار لحظة.")
        updates["final_response"] = final_response
        new_history = state.get("chat_history", []) + [{"role":"agent","content": final_response}]
        updates["chat_history"] = new_history
        return updates
    context_parts = []
    if state.get("sql_result"):
        context_parts.append("SQL results: " + json.dumps(state["sql_result"], ensure_ascii=False))
    if state.get("retrieved_context"):
        context_parts.append("Knowledge: " + " || ".join(state["retrieved_context"]))
    recent = "\n".join([f"{m['role']}: {m['content']}" for m in state.get("chat_history",[])[-6:]])
    context = "\n".join(context_parts + [recent])
    prompt = RESPONSE_PROMPT_TEMPLATE.format(context=context, user_message=state.get("extracted_text",""), emotion=state.get("emotion",""), intent=state.get("intent",""), response_item = r)
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role":"user","content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        out = resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("Response generation failed: %s", e)
        out = "عذرًا، حدث خطأ أثناء تجهيز الرد. سأحوّل المحادثة لموظف خدمة العملاء."
        updates["escalate"] = True
    if not r:
      out = make_response_item("text", out)
    else:
      out = {**r, "message": out}
    updates["final_response"] = out
    new_history = state.get("chat_history", []) + [{"role":"agent","content": out}]
    updates["chat_history"] = new_history

    return updates

def task_router(state):
    task = state.get("pending_task")
    return task

# ---------------- Build LangGraph with correct conditional routing ----------------
workflow = StateGraph(MuttammState)

# Core perception & understanding nodes
workflow.add_node("detect_input_type", detect_input_type_node)
workflow.add_node("speech_to_text", speech_to_text_node)
workflow.add_node("vision_analyzer", vision_analyzer_node)
workflow.add_node("emotion_detection", emotion_detection_node)
workflow.add_node("pre_escalation_check", pre_escalation_check_node)
workflow.add_node("intent_classification", intent_classification_node)

# Main router node
workflow.add_node("router", router_node)

# Sub-agents
workflow.add_node("ahwal_agent", ahwal_subagent_node)
workflow.add_node("jawazat_agent", jawazat_subagent_node)
workflow.add_node("murur_agent", murur_subagent_node)
workflow.add_node("wafeedin_agent", wafeedin_subagent_node)
workflow.add_node("other_agent", other_subagent_node)

# Post handler & response
workflow.add_node("post_handler", post_handler_node)
workflow.add_node("response_generator", response_generator_node)


# ---------------- Detect input type router ----------------
def detect_input_router(state: MuttammState):
    t = state.get("message_type")
    if t == "voice":
        return "voice"
    if t == "image":
        return "image"
    return "text"


workflow.add_conditional_edges(
    "detect_input_type",
    detect_input_router,
    {
        "voice": "speech_to_text",
        "image": "vision_analyzer",
        "text": "emotion_detection",
    }
)

workflow.add_edge("speech_to_text", "emotion_detection")
workflow.add_edge("vision_analyzer", "emotion_detection")
workflow.add_edge("emotion_detection", "pre_escalation_check")
workflow.add_edge("pre_escalation_check", "intent_classification")
workflow.add_edge("intent_classification", "router")


# ---------------- Main router for Muttamm ----------------
workflow.add_conditional_edges(
    "router",
    task_router,
    {
        "الأحوال المدنية": "ahwal_agent",
        "الجوازات": "jawazat_agent",
        "المرور": "murur_agent",
        "شؤون الوافدين": "wafeedin_agent",
        "Other": "other_agent",
        "escalate": "response_generator",
    }
)


# ---------------- Sub-agent → Post handler ----------------
workflow.add_edge("ahwal_agent", "post_handler")
workflow.add_edge("jawazat_agent", "post_handler")
workflow.add_edge("murur_agent", "post_handler")
workflow.add_edge("wafeedin_agent", "post_handler")
workflow.add_edge("other_agent", "post_handler")


# ---------------- Post handler → Response generator ----------------
workflow.add_edge("post_handler", "response_generator")
workflow.add_edge("response_generator", END)


# ---------------- Entry point ----------------
workflow.set_entry_point("detect_input_type")
agent_app = workflow.compile()

def run_agent(message, state=None):
    if state is None:
        state = {}
    state["user_current_message"] = message
    result = agent_app.invoke(state)
    return result