import streamlit as st
import json
from agent_engine import run_agent

# ===============================================================
# 2) SESSION STATE
# ===============================================================

if "state" not in st.session_state:
    st.session_state.state = {}          # FULL AGENT STATE

if "chat" not in st.session_state:
    st.session_state.chat = []           # list of {"role": "user/agent", "content": str}

if "last_response" not in st.session_state:
    st.session_state.last_response = None

# ===============================================================
# 3) RENDER CHAT BUBBLES
# ===============================================================

def bot_msg(text):
    st.markdown(
        f"""
        <div style="background:#f1f5f9;padding:10px 14px;border-radius:18px 18px 18px 4px;
                    font-size:15px;margin-bottom:8px;max-width:80%;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

def user_msg(text):
    st.markdown(
        f"""
        <div style="background:#16a34a;color:white;padding:10px 14px;border-radius:18px 18px 4px 18px;
                    font-size:15px;margin-bottom:8px;max-width:80%;margin-right:auto;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===============================================================
# 4) RENDER AGENT RESPONSE (options, confirm, upload…)
# ===============================================================

def render_agent_response(resp):
    """
    resp is: final_response dict
    {
       "type": "text" | "options" | "confirm" | "payment" | "upload",
       "message": "...",
       "options": [...],
       "input_key": ...,
       "metadata": {...}
    }
    """

    msg_type = resp.get("type", "text")
    text = resp.get("message", "")
    options = resp.get("options", [])
    metadata = resp.get("metadata", {})

    # ------------------------------
    # CASE 1: TEXT ONLY
    # ------------------------------
    if msg_type == "text" or msg_type == "done":
        return

    # ------------------------------
    # CASE 2: OPTIONS
    # ------------------------------
    if msg_type == "options":
        for opt in options:
            if st.button(opt, key=f"opt_{opt}"):
                # Treat option click as a NEW user message
                st.session_state.chat.append({"role": "user", "content": opt})
                # Run agent again using same state
                result = run_agent(opt, st.session_state.state)
                st.session_state.last_response = result["final_response"]
                st.session_state.chat.append({"role": "agent", "content": result["final_response"]["message"]})
                st.session_state.state = result
                st.rerun()

    # ------------------------------
    # CASE 3: CONFIRMATION
    # ------------------------------
    if msg_type == "confirm":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("نعم"):
                st.session_state.chat.append({"role": "user", "content": "نعم"})
                result = run_agent("نعم", st.session_state.state)
                st.session_state.last_response = result["final_response"]
                st.session_state.chat.append({"role": "agent", "content": result["final_response"]["message"]})
                st.session_state.state = result
                st.rerun()

        with col2:
            if st.button("لا"):
                st.session_state.chat.append({"role": "user", "content": "لا"})
                result = run_agent("لا", st.session_state.state)
                st.session_state.last_response = result["final_response"]
                st.session_state.chat.append({"role": "agent", "content": result["final_response"]["message"]})
                st.session_state.state = result
                st.rerun()

    # ------------------------------
    # CASE 4: PAYMENT
    # ------------------------------
    if msg_type == "payment":
        amount = metadata.get("amount", "?")
        st.markdown(f"**المبلغ المطلوب للدفع: {amount} ريال**")

        if st.button("ادفع الآن"):
            st.session_state.chat.append({"role": "user", "content": "تم الدفع"})
            result = run_agent("تم الدفع", st.session_state.state)
            st.session_state.last_response = result["final_response"]
            st.session_state.chat.append({"role": "agent", "content": result["final_response"]["message"]})
            st.session_state.state = result
            st.rerun()

    # ------------------------------
    # CASE 5: FILE UPLOAD
    # ------------------------------
    if msg_type == "upload":
        uploaded = st.file_uploader("ارفع الملف هنا:")
        if uploaded:
            user_text = f"تم رفع الملف: {uploaded.name}"
            st.session_state.chat.append({"role": "user", "content": user_text})

            result = run_agent(user_text, st.session_state.state)
            st.session_state.last_response = result["final_response"]
            st.session_state.chat.append({"role": "agent", "content": result["final_response"]["message"]})
            st.session_state.state = result
            st.rerun()



# ===============================================================
# 5) MAIN CHAT UI
# ===============================================================

def chat_engine():
    st.title("متمم – مساعدك الذكي")

    # 1. DISPLAY CHAT HISTORY
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            user_msg(msg["content"])
        else:
            bot_msg(msg["content"])

    # ---
    
    # 2. RENDER INTERACTIVE RESPONSE (Buttons, Uploader, etc.)
    if st.session_state.last_response:
        resp = st.session_state.last_response
        msg_type = resp.get("type", "text")


        if msg_type in ["options", "confirm", "payment", "upload"]:
            # Show message
            bot_msg(resp.get("message", ""))
            # Render the interactive part (buttons, uploader)
            render_agent_response(resp) 
            
            # Use return here to stop the script. The interaction widgets
            return
    
    # ---

    # 3. MAIN TEXT INPUT FORM
    st.markdown("---")
    
    with st.form(key="user_input_form", clear_on_submit=True):
        user_input = st.text_input("اكتب رسالتك…", key="user_input_box", label_visibility="collapsed")
        submit_button = st.form_submit_button("إرسال")

        if submit_button and user_input:
            # 1. Add user message
            st.session_state.chat.append({"role": "user", "content": user_input})

            # 2. Clear any pending interactive response before running the new query
            st.session_state.last_response = None 

            # 3. Run agent
            result = run_agent(user_input, st.session_state.state)

            # 4. Save the full state and final response
            final_response = result["final_response"]
            st.session_state.state = result
            
            # 5. Set the final response for the next run's rendering
            st.session_state.last_response = final_response
            
            # 6. Add agent's response text to chat history
            st.session_state.chat.append({"role": "agent", "content": final_response["message"]})
            
            # 7. Rerun to display the history and potentially the interactive elements
            st.rerun()

