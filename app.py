import streamlit as st
import base64
from chat_engine import chat_engine

# إعداد عام للصفحة
st.set_page_config(
    page_title="منسق أبشر الذكي",
    layout="wide"
)

# ===== حالة الجلسة =====
if "m_screen" not in st.session_state:
    st.session_state.m_screen = "home"    
if "passport_photo_uploaded" not in st.session_state:
    st.session_state.passport_photo_uploaded = False
if "m_output_type" not in st.session_state:
    st.session_state.m_output_type = "text"   # text / options / upload
if "menu_choice" not in st.session_state:
    st.session_state.menu_choice = "منسق أبشر الذكي"
if "m_screen" not in st.session_state:
    st.session_state.m_screen = "home"
if "passport_photo_uploaded" not in st.session_state:
    st.session_state.passport_photo_uploaded = False
if "menu_choice" not in st.session_state:
    st.session_state.menu_choice = "منسق أبشر الذكي"
if "m_output_type" not in st.session_state:
    st.session_state.m_output_type = "text"
if "m_options_category" not in st.session_state:
    st.session_state.m_options_category = None
if "m_options_service" not in st.session_state:
    st.session_state.m_options_service = None


# ===== CSS عام (الهيدر + أبشر + متمم + السايدبار) =====
st.markdown(
    """
    <style>
    * {
        font-family: "Tahoma", sans-serif;
    }

    html, body, [data-testid="stAppViewContainer"] {
        direction: rtl;
        background-color: #f4f7f6;
    }

    /* الهيدر الأساسي */
    .header-bar {
        width: 100%;
        background: #ffffff;
        padding: 8px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid #e3e7e5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
        box-sizing: border-box;
    }
    .header-left img { height: 60px; }
    .header-center { display: flex; gap: 10px; }
    .header-card {
        background: #ffffff;
        border: 1px solid #e1e1e1;
        border-radius: 10px;
        padding: 8px 14px;
        display: flex;
        flex-direction: column;
        align-items: center;
        font-size: 12px;
        width: 95px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .header-card img { height: 22px; margin-bottom: 4px; }
    .header-right img { height: 60px; margin-left: 18px; }

    /* شريط منسق أبشر */
    .top-bar {
        background-color: #006c35;
        color: #ffffff;
        padding: 10px 18px;
        border-radius: 0 0 16px 16px;
        margin-bottom: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
    }
    .top-bar-title { font-weight: bold; font-size: 18px; }
    .top-bar-subtitle { font-size: 13px; opacity: 0.95; }
    .top-bar-user { font-size: 13px; text-align: left; }

    /* كروت عامة */
    .card {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 16px;
        border: 1px solid #e3e7e5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        margin-bottom: 12px;
    }

    /* كروت الخدمات الاستباقية */
    .pro-card {
        background-color:#ffffff;
        border:1px solid #e3e7e5;
        border-radius:24px;
        padding:20px 14px 16px 14px;
        text-align:center;
        box-shadow:0 2px 6px rgba(0,0,0,0.06);
        margin-bottom:14px;
    }

    /* أزرار أبشر العامة (خارج متمم) */
    .stButton > button {
        background-color: #00a86b !important;
        color: white !important;
        border-radius: 20px !important;
        border: none !important;
        padding: 0.35rem 1.2rem !important;
        font-size: 14px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #008853 !important;
    }

    /* سايدبار */
    .menu-wrapper { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }

    .menu-item .stButton > button,
    .menu-item-active .stButton > button {
        background-color: #ffffff !important;
        color: #222 !important;
        border-radius: 10px !important;
        border: 1px solid #e4e7e6 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
        padding: 10px 12px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        text-align: right !important;
        justify-content: flex-start !important;
    }
    .menu-item-active .stButton > button {
        background-color: #e7f5ee !important;
        border-color: #00a86b !important;
        border-right: 4px solid #00a86b !important;
        color: #006c47 !important;
        font-weight: 600 !important;
    }

    /* نصوص داخل الكروت */
    .card h3, .card h4, .card h5, .card p, .card span, .card li {
        color: #233239;
    }
    hr { border: none; border-top: 1px solid #e3e7e5; margin: 8px 0; }

    /* ===== متمم (داخل #mtamem-wrapper فقط) ===== */

    /* نخلي كل شي يمين */
    #mtamem-wrapper,
    #mtamem-wrapper * {
        text-align: right !important;
        direction: rtl !important;
    }

    /* فقاعات الشات */
    #mtamem-wrapper .msg-bot {
        background: #f9fafb;
        border-radius: 18px 18px 18px 4px;
        padding: 10px 14px;
        font-size: 15px;
        color: #111827;
        max-width: 80%;
        margin-bottom: 10px;
    }
    #mtamem-wrapper .msg-user {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        border-radius: 18px 18px 4px 18px;
        padding: 10px 14px;
        font-size: 15px;
        color: #ffffff;
        max-width: 80%;
        margin-bottom: 10px;
        margin-right: auto;
        box-shadow: 0 4px 10px rgba(22, 163, 74, 0.25);
    }

    /* ترويسة متمم */
    #mtamem-wrapper .chat-header-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 16px;
        margin-bottom: 12px;
    }

    /* أزرار متمم – كابسولات ناعمة */
    #mtamem-wrapper .stButton > button {
        background: #f4f5ff !important;
        color: #111827 !important;
        border-radius: 999px !important;
        border: 1px solid #d4ddff !important;
        box-shadow: 0 4px 10px rgba(99, 102, 241, 0.12) !important;
        padding: 0.45rem 1.3rem !important;
        font-size: 14px !important;
        font-weight: 600 !important;
    }
    #mtamem-wrapper .stButton > button:hover {
        background: #e0e7ff !important;
        border-color: #6366f1 !important;
    }

    /* شريط الإدخال في أسفل متمم */
    #mtamem-wrapper .chat-input-row {
        margin-top: 16px;
        padding-top: 8px;
        border-top: 1px solid #e5e7eb;
    }

    #mtamem-wrapper .chat-input-inner {
        background: #ffffff;
        border-radius: 999px;
        border: 1px solid #e5e7eb;
        padding: 6px 10px;
        display: flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
    }

    /* نخلي التكست إنبت (وكل حقول النص) من اليمين */
    #mtamem-wrapper [data-testid="stTextInput"] input,
    #mtamem-wrapper [data-testid="stTextArea"] textarea,
    #mtamem-wrapper [data-testid="stChatInput"] textarea,
    #mtamem-wrapper input[type="text"] {
        text-align: right !important;
        direction: rtl !important;
    }

    /* الأيقونات الصغيرة تحت (منيو – فويس – رفع) */
    #mtamem-wrapper .icon-button .stButton > button {
        background:#ffffff !important;
        border-radius:999px !important;
        border:1px solid #e5e7eb !important;
        box-shadow:0 2px 6px rgba(15,23,42,0.06) !important;
        padding:0.25rem 0.6rem !important;
        font-size:16px !important;
        width:100% !important;
        min-width:0 !important;
    }
    #mtamem-wrapper .icon-button .stButton > button:hover {
        background:#f3f4f6 !important;
        border-color:#9ca3af !important;
    }

    /* جعل عناوين التبويبات يمين */
    [data-baseweb="tab-panel"] h1,
    [data-baseweb="tab-panel"] h2,
    [data-baseweb="tab-panel"] h3,
    [data-baseweb="tab-panel"] .stMarkdown h1,
    [data-baseweb="tab-panel"] .stMarkdown h2,
    [data-baseweb="tab-panel"] .stMarkdown h3 {
        text-align: right !important;
    }

    /* جعل الجملة التوضيحية تحت العنوان يمين */
    [data-baseweb="tab-panel"] p {
        text-align: right !important;
    }

   /* جعل عنوان تبويب "متمم، مساعدك الشخصي" يمين */
    [data-baseweb="tab"] {
        text-align: right !important;
        direction: rtl !important;
    }

    /* جعل النص داخل محتوى التاب يمين */
    [data-baseweb="tab-panel"] * {
        text-align: right !important;
        direction: rtl !important;
    }

    /* إعادة محاذاة كروت الخدمات الاستباقية للوسط فقط */
    .pro-card,
    .pro-card * {
        text-align: center !important;
    }

    [

    </style>
    """,
    unsafe_allow_html=True
)


def img_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ===== الهيدر العام الثابت =====
def global_header():
    ministry_b64 = img_to_base64("Ministry_of_Interior_Saudi_Arabia.png")
    vision_b64 = img_to_base64("saudi-vision-2030.png")
    absher_b64 = img_to_base64("logo_absher.png")

    st.markdown(
        f"""
<div class="header-bar">
  <div class="header-left">
    <img src="data:image/png;base64,{ministry_b64}">
  </div>
  <div class="header-center">
    <div class="header-card">
      <img src="https://cdn-icons-png.flaticon.com/512/1828/1828479.png">
      <div>تسجيل الخروج</div>
    </div>
    <div class="header-card">
      <img src="https://cdn-icons-png.flaticon.com/512/197/197484.png">
      <div>English</div>
    </div>
    <div class="header-card">
      <img src="https://cdn-icons-png.flaticon.com/512/1828/1828778.png">
      <div>دليل الخدمات</div>
    </div>
    <div class="header-card">
      <img src="https://cdn-icons-png.flaticon.com/512/1827/1827310.png">
      <div>الإشعارات</div>
    </div>
    <div class="header-card">
      <img src="https://cdn-icons-png.flaticon.com/512/3524/3524636.png">
      <div>تعديل المستخدم</div>
    </div>
    <div class="header-card">
      <img src="https://cdn-icons-png.flaticon.com/512/1828/1828614.png">
      <div>لوحة المعلومات</div>
    </div>
  </div>
  <div class="header-right">
    <img src="data:image/png;base64,{vision_b64}">
    <img src="data:image/png;base64,{absher_b64}">
  </div>
</div>
        """,
        unsafe_allow_html=True
    )

# ===== دالة كارد الخدمات الاستباقية (2×2) =====
def render_pro_card(icon, title, subtitle, btn_label, btn_key):
    with st.container():
        st.markdown(
            f"""
            <div class="pro-card">
                <div style="font-size:36px;margin-bottom:10px;">{icon}</div>
                <div style="font-size:15px;font-weight:700;color:#233239;margin-bottom:4px;">
                    {title}
                </div>
                <div style="font-size:13px;color:#4d5a60;margin-bottom:12px;">
                    {subtitle}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        left, center, right = st.columns([1, 2, 1])
        with center:
            st.button(btn_label, key=btn_key)

# ===== شريط منسق أبشر الذكي =====
def coordinator_bar():
    st.markdown(
        """
        <div class="top-bar">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:16px;">
                <div class="top-bar-title">منسق أبشر الذكي</div>
                <div class="top-bar-subtitle">
                    وكيل رقمي يساعدك على متابعة وتجديد وثائقك في الوقت المناسب، ويجمع خدماتك في مكان واحد ✨
                </div>
                <div class="top-bar-user">مرحباً، سديم 👤</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===== صفحة منسق أبشر الذكي =====
def page_coordinator():
    coordinator_bar()
    tab_status, tab_proactive, tab_mtamem = st.tabs(
        ["حالة الطلب", "خدمات استباقية", "متمم، مساعدك الشخصي"]
    )

    # --- تبويب حالة الطلب ---
    with tab_status:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("حالة الطلب – تجديد الجواز")
        st.write("تابع خطوات تنفيذ طلب تجديد الجواز السعودي:")

        steps = [
            ("التحقق من الجواز", "pending"),
            ("السداد", "pending"),
            ("التحقق من المستندات", "processing"),
            ("تم إرسال الطلب", "done"),
        ]

        cols = st.columns(len(steps))
        for i, (label, status) in enumerate(steps):
            with cols[i]:
                if status == "done":
                    icon = "🟢"
                elif status == "processing":
                    icon = "🟠"
                else:
                    icon = "⚪"

                st.markdown(
                    f"<div style='text-align:center;font-size:26px;'>{icon}</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div style='text-align:center;font-size:13px;'>{label}</div>",
                    unsafe_allow_html=True,
                )

        total_steps = 4
        current_step = 2
        progress_value = current_step / total_steps
        st.progress(progress_value)

        st.info("يعمل متمم الآن على إكمال الخطوات (التحقق من الجواز)، وسيتم إشعارك فور الانتهاء.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- تبويب الخدمات الاستباقية ---
    with tab_proactive:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("خدمات استباقية")
        st.write("راجعنا بياناتك ووجدنا الخدمات التالية التي تحتاج لاهتمامك:")

        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            render_pro_card("🛂", "جواز السفر السعودي",
                            "تبقّى 7 أشهر على انتهاء جواز السفر.",
                            "تجديد الجواز", "pro_passport")
        with row1_col2:
            render_pro_card("👮‍♂️", "رخصة القيادة الخاصة",
                            "تنتهي خلال شهر — يُقترح التجديد مبكّرًا.",
                            "تجديد رخصة", "pro_license")

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            render_pro_card("🚘", "استمارة مركبة سوناتا 2018",
                            "تبقّى 6 أشهر — يمكنك جدولة التجديد مبكرًا.",
                            "تجديد استمارة", "pro_estmara")
        with row2_col2:
            render_pro_card("🛡️", "تأمين مركبة: كامري",
                            "تبقّى 9 أشهر — يمكن تفعيل تذكير قبل الانتهاء.",
                            "تجديد التأمين", "pro_insurance")

        st.markdown("---")
        st.info("يمكن تفعيل هذه الخدمات أيضًا من داخل محادثة متمم.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- تبويب متمم، مساعدك الشخصي ---
    with tab_mtamem:
        st.markdown('<div id="mtamem-wrapper">', unsafe_allow_html=True)
        chat_engine()   # <-- call the chat function
        st.markdown("</div>", unsafe_allow_html=True)

   
# ===== صفحات باقي القائمة =====
def placeholder_page(title):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title(title)
    st.info("هذه صفحة تجريبية لعرض الفكرة فقط، التركيز الرئيسي على منسق أبشر الذكي ومتمم.")
    st.markdown("</div>", unsafe_allow_html=True)

# ===== تخطيط الصفحة: هيدر عام + محتوى + سايدبار يمين =====
global_header()

col_main, col_menu = st.columns([4, 1])

with col_menu:
    st.markdown("""
        <style>
        .menu-box {
            background:#ffffff;
            border:1px solid #e4e7e6;
            border-radius:12px;
            overflow:hidden;
            box-shadow:0 1px 3px rgba(0,0,0,0.05);
        }
        .menu-item {
            padding:14px;
            font-size:14px;
            border-bottom:1px solid #f1f1f1;
            cursor:pointer;
            display:flex;
            align-items:center;
            gap:10px;
            transition:0.2s;
        }
        .menu-item:hover {
            background:#f7faf8;
        }
        .menu-item span.icon {
            font-size:18px;
            opacity:0.7;
        }
        .menu-item-active {
            background:#e7f5ee;
            border-right:5px solid #00a86b;
            font-weight:bold;
            color:#006c47;
        }
        .menu-item-active span.icon {
            color:#00a86b !important;
            opacity:1 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    menu_options = {
        "الخدمات الإلكترونية": "🖥️",
        "التفاويض": "🤝",
        "استبيانات أبشر": "📄",
        "المدفوعات الحكومية": "💰",
        "منسق أبشر الذكي": "🤖"
    }

    st.markdown("<div class='menu-box'>", unsafe_allow_html=True)

    for label, icon in menu_options.items():
        active = (st.session_state.menu_choice == label)
        css_class = "menu-item-active" if active else "menu-item"

        if st.button(f"{icon}  {label}", key=f"btn_{label}", use_container_width=True):
            st.session_state.menu_choice = label
            st.rerun()

        st.markdown(f"""
            <div class="{css_class}">
                <span class="icon">{icon}</span> {label}
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("إعادة ضبط متمم"):
        st.session_state.m_output_type = "text"
        st.success("تمت إعادة ضبط متمم.")

with col_main:
    choice = st.session_state.menu_choice

    if choice.startswith("منسق أبشر الذكي"):
        page_coordinator()
    else:
        placeholder_page(choice.split(" ")[0])
