import streamlit as st
import os
import json
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px
from PIL import Image
from dotenv import load_dotenv
from agents import PharmaGuardMAS
from utils import setup_rag, generate_pdf_report
from tts_utils import transcribe_audio, text_to_speech, get_audio_html

load_dotenv()

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def get_secret(key, default=""):
    """Sırları sırasıyla st.secrets, os.getenv ve default değerden çeker. Placeholder değerleri atlar."""
    val = default
    try:
        # 1. Önce Streamlit Secrets (Cloud veya secrets.toml)
        val = st.secrets.get(key, "")
        # Eğer placeholder bir değerse (örneğin 'your_...') veya boşsa os.getenv'e bak
        if not val or val.lower().startswith("your_"):
            val = os.getenv(key, default)
    except Exception:
        # st.secrets erişilemezse (lokal çalıştırma vb) env'e bak
        val = os.getenv(key, default)
    
    return val.strip() if isinstance(val, str) else val


def make_gauge(score: int) -> go.Figure:
    color = "#22c55e" if score < 30 else "#f59e0b" if score < 65 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Risk Skoru", "font": {"color": "#94a3b8", "size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 30], "color": "rgba(34,197,94,0.1)"},
                {"range": [30, 65], "color": "rgba(245,158,11,0.1)"},
                {"range": [65, 100], "color": "rgba(239,68,68,0.1)"},
            ],
            "threshold": {"line": {"color": color, "width": 4}, "thickness": 0.75, "value": score},
        },
        number={"suffix": "/100", "font": {"color": color, "size": 28}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f8fafc"}, height=220, margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Akıllı Eczane | AI Sağlık Asistanı",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@300;600;900&display=swap" rel="stylesheet">
<style>
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #1e293b 40%, #020617 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .agent-card {
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.4);
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    }
    .agent-card:hover {
        border: 1px solid rgba(59,130,246,0.4);
        transform: translateY(-2px);
    }
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 900;
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 0;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in { animation: fadeInUp 0.8s ease-out forwards; }
    .red-alarm {
        background: linear-gradient(90deg, rgba(239,68,68,0.1) 0%, rgba(239,68,68,0.2) 100%);
        border-left: 5px solid #ef4444;
        color: #fca5a5;
        padding: 18px 22px;
        border-radius: 12px;
        font-weight: 600;
        margin: 16px 0;
        box-shadow: 0 4px 15px rgba(239,68,68,0.1);
    }
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255,255,255,0.05);
    }
    .engine-tag {
        background: rgba(59,130,246,0.1);
        color: #60a5fa;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        border: 1px solid rgba(59,130,246,0.2);
    }
    h1, h2, h3 { font-family: 'Outfit', sans-serif; font-weight: 600; }

    /* Chat bubbles */
    .chat-user {
        background: rgba(59,130,246,0.15);
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 14px 14px 4px 14px;
        padding: 12px 16px;
        margin: 8px 0 8px 30%;
        color: #e0f2fe;
    }
    .chat-bot {
        background: rgba(168,85,247,0.12);
        border: 1px solid rgba(168,85,247,0.25);
        border-radius: 14px 14px 14px 4px;
        padding: 12px 16px;
        margin: 8px 30% 8px 0;
        color: #f5d0fe;
    }

    /* Section badge */
    .section-badge {
        display: inline-block;
        background: linear-gradient(135deg,#1e3a5f,#1e1b4b);
        border: 1px solid rgba(99,102,241,0.3);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8em;
        color: #a5b4fc;
        margin-bottom: 10px;
    }
    
    /* Report section cards */
    .report-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 16px 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── INIT SESSION STATE ────────────────────────────────────────────────────────
if 'vector_db' not in st.session_state:
    with st.spinner("🏥 Tıbbi Hafıza Hazırlanıyor..."):
        st.session_state.vector_db = setup_rag()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'report' not in st.session_state:
    st.session_state.report = None

if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False

if 'risk_score' not in st.session_state:
    st.session_state.risk_score = None

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("assets/pharmacy_logo.png", width=110)
    st.title("Kontrol Merkezi")

    if st.button("🔄 Sistemi Sıfırla", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    # API Keys
    g_key = str(get_secret("GOOGLE_API_KEY", "")).strip()
    gr_key = str(get_secret("GROQ_API_KEY", "")).strip()

    if g_key and gr_key:
        try:
            if ('mas_v2' not in st.session_state
                    or st.session_state.get('last_g_key') != g_key
                    or st.session_state.get('last_gr_key') != gr_key):
                st.session_state.mas_v2      = PharmaGuardMAS(g_key, gr_key)
                st.session_state.last_g_key  = g_key
                st.session_state.last_gr_key = gr_key
            st.success("✅ Sistem Hazır")
            with st.expander("🔍 Sistem Detayları"):
                st.write("**👁 Vision:** Gemini + Groq Fallback")
                st.write("**📚 RAG:** Local HuggingFace")
                st.write("**🌐 Web:** DuckDuckGo Search")
                st.write("**🔊 TTS:** gTTS (Türkçe)")
        except Exception as e:
            st.error(f"Sistem başlatılamadı: {e}")

    st.markdown("---")
    # ── KİŞİSEL SAĞLIK PROFİLİ ────────────────────────────────────────────
    st.markdown("#### 👤 Kişisel Sağlık Profili")
    with st.expander("Profilimi Düzenle", expanded=False):
        u_name     = st.text_input("Adınız", placeholder="Ali İhsan", key="u_name")
        u_age      = st.number_input("Yaşınız", min_value=1, max_value=120, value=30, key="u_age")
        u_diseases = st.multiselect(
            "Kronik Hastalıklarınız",
            ["Kalp Hastalığı", "Diyabet (Şeker)", "Hipertansiyon", "Astım", "Böbrek Hastalığı",
             "Karaciğer Hastalığı", "Tiroid", "Epilepsi", "KOAH", "Kanser"],
            key="u_diseases"
        )
        u_allergies  = st.text_area("Alerjileriniz", placeholder="Penisilin, Polen...", key="u_allergies", height=60)
        u_curr_meds  = st.text_area("Kullandığınız İlaçlar", placeholder="Metformin 500mg...", key="u_curr_meds", height=60)

    user_profile = {
        "name": st.session_state.get("u_name", ""),
        "age":  st.session_state.get("u_age",  30),
        "diseases":    st.session_state.get("u_diseases", []),
        "allergies":   st.session_state.get("u_allergies", ""),
        "current_meds": st.session_state.get("u_curr_meds", ""),
    }

    st.markdown("---")
    # ── SESLİ ASISTAN ─────────────────────────────────────────────────────
    st.markdown("#### 🎙️ Sesli Asistan")
    audio_in = st.audio_input("Sorunuzu konuşarak sorun:", key="voice_input")
    if audio_in and st.session_state.get('report') and gr_key:
        with st.spinner("🎧 Ses çözümleniyor..."):
            audio_bytes  = audio_in.read()
            question_stt = transcribe_audio(audio_bytes, st.session_state.mas_v2.groq_client)
        if question_stt and "hatası" not in question_stt.lower():
            st.info(f"🗣️ *{question_stt}*")
            with st.spinner("🤔 Cevap hazırlanıyor..."):
                stt_answer = st.session_state.mas_v2.chat_with_report(
                    question_stt,
                    st.session_state.report,
                    st.session_state.chat_history
                )
            st.markdown(f"**💬 Cevap:** {stt_answer}")
            tts_bytes = text_to_speech(stt_answer)
            if tts_bytes:
                st.markdown(get_audio_html(tts_bytes), unsafe_allow_html=True)
        else:
            st.warning(question_stt)

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center;color:#94a3b8;font-size:0.9em;">'
        '👨‍💻 Tasarım & Geliştirme:<br>'
        '<b style="color:#60a5fa;">Ali İhsan ÇETİN</b>'
        '</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN AREA — HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title fade-in">Akıllı Eczane</div>', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;color:#94a3b8;margin-bottom:0.25rem;">'
    'Yapay Zeka Destekli Çok Ajanlı İlaç Analiz Platformu</div>',
    unsafe_allow_html=True
)

# ─── SECRETS VALIDATION (Dinamik Kontrol) ─────────────────────────────────
# Sidebar'daki g_key ve gr_key değerlerini de kontrol ediyoruz
is_g_empty = (not g_key or g_key.lower().startswith("your_"))
is_gr_empty = (not gr_key or gr_key.lower().startswith("your_"))

st.markdown(
    '<div style="text-align:center;color:#64748b;margin-bottom:1.8rem;font-size:0.88em;">'
    '👨‍💻 Geliştirici: Ali İhsan ÇETİN</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────────────
# COLUMNS — UPLOAD | ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1.6], gap="large")

# ── SOL KOLON — GÖRSEL YÜKLEYİCİ ──────────────────────────────────────────
with col1:
    st.markdown('<div class="agent-card fade-in">', unsafe_allow_html=True)
    st.markdown("### 📸 Görsel Tarayıcı")
    st.caption("İlaç kutusu + prospektüs fotoğrafı yükleyebilirsiniz (çoklu seçim)")
    img_files = st.file_uploader(
        "Görselleri buraya sürükleyin...",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        key="img_uploader"
    )
    if img_files:
        for i, f in enumerate(img_files):
            st.image(f, use_container_width=True,
                     caption=f"{'📦 İlaç Kutusu' if i == 0 else '📄 Prospektüs'}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Risk Gauge (analiz sonrası)
    if st.session_state.risk_score is not None:
        st.markdown('<div class="agent-card fade-in">', unsafe_allow_html=True)
        st.markdown("### 🎯 Risk Göstergesi")
        st.plotly_chart(make_gauge(st.session_state.risk_score), use_container_width=True)
        score = st.session_state.risk_score
        if score < 30:
            st.success("🟢 Düşük Risk")
        elif score < 65:
            st.warning("🟡 Orta Risk — Dikkatli Kullanın")
        else:
            st.error("🔴 Yüksek Risk — Doktora Danışın!")
        st.markdown('</div>', unsafe_allow_html=True)

# ── SAĞ KOLON — ANALİZ & RAPOR ────────────────────────────────────────────
with col2:
    if img_files and g_key and gr_key:
        # Streamlit her rerun'da analizi tekrar çalıştırmaması için state kontrolü
        img_keys = tuple(f.name + str(f.size) for f in img_files)
        if st.session_state.get('last_img_keys') != img_keys:
            st.session_state.analysis_done = False
            st.session_state.report = None
            st.session_state.chat_history = []
            st.session_state.risk_score = None

        if not st.session_state.analysis_done:
            st.markdown('<div class="agent-card fade-in">', unsafe_allow_html=True)
            st.markdown("### 🔬 Canlı Analiz Süreci")
            
            with st.status("🔍 Akıllı Eczane Ajanları Çalışıyor...", expanded=True) as status:
                try:
                    mas = st.session_state.mas_v2
                    image_bytes_list = [f.getvalue() for f in img_files]
                    
                    # 1. Vision Scanner
                    st.write("📸 Görsel Analiz ediliyor...")
                    v_data = mas.vision_scanner(image_bytes_list)
                    if "error" in v_data:
                        st.error(f"Görsel Analiz Hatası: {v_data['error']}")
                        status.update(label="❌ Analiz Durduruldu", state="error")
                        st.stop()
                    st.caption(f"Aktif Motor: {mas.vision_engine}")

                    drug = (v_data.get("Ilac_Adi") or v_data.get("Ilac Adi")
                            or v_data.get("İlaç Adı") or v_data.get("Drug Name", "Bilinmeyen"))

                    st.write("🚀 **Dağıtık Analiz Başlatıldı** (Multi-threading)...")
                    
                    # 1. Aşama Paralel Görevler (Web Arama, Kurumsal Analiz ve RAG)
                    with ThreadPoolExecutor(max_workers=3) as executor:
                        st.write("🔄 Web, Kurumsal ve RAG ajanları eşzamanlı çalışıyor...")
                        future_web = executor.submit(mas.web_search, drug)
                        future_corp = executor.submit(mas.corporate_analyst, str(v_data))
                        
                        # RAG ve Safety birbirine bağlı olduğu için aynı thread içinde sıralı zincirliyoruz.
                        def rag_safety_chain(v_db, usr_prof):
                            r_data = mas.rag_specialist(drug, v_db)
                            saf_data = mas.safety_auditor(r_data, usr_prof)
                            return r_data, saf_data
                        
                        v_db_ref = st.session_state.get('vector_db')
                        future_rag_saf = executor.submit(rag_safety_chain, v_db_ref, user_profile)
                        
                        web_data = future_web.result()
                        c_data = future_corp.result()
                        rag_data, s_data = future_rag_saf.result()

                    # 2. Aşama Paralel Görevler (Eczacı ve Risk Skoru)
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        st.write("🔄 Alternatifler ve Risk Skoru hesaplanıyor...")
                        future_pharm = executor.submit(mas.pharmacist_agent, s_data, v_data, user_profile)
                        future_risk = executor.submit(mas.extract_risk_score, s_data)

                        pharm_data = future_pharm.result()
                        risk_score = future_risk.result()
                        st.session_state.risk_score = risk_score

                    st.write("✍️ **Report-Synthesizer:** Rapor yazılıyor...")
                    report = mas.synthesize_report(
                        v_data, rag_data, s_data, c_data,
                        web_data, pharm_data, user_profile
                    )
                    st.caption(f"📢 **Son İşlem Modeli:** {mas.last_used_model}")
                    st.session_state.report = report
                    st.session_state.last_img_keys = img_keys
                    st.session_state.analysis_done = True
                    status.update(label=f"✅ Analiz Tamamlandı! ({mas.last_used_model})", state="complete")
                except Exception as e:
                    st.error(f"Sistem Hatası: {e}")
                    status.update(label="❌ Kritik Hata!", state="error")
            
            st.markdown('</div>', unsafe_allow_html=True)

        # ── RAPOR GÖSTERİMİ ───────────────────────────────────────────────────
        if st.session_state.analysis_done and st.session_state.report:
            report = st.session_state.report

            # Kırmızı Alarm Banner
            if "KIRMIZI ALARM" in report.upper():
                st.markdown(
                    '<div class="red-alarm">⚠️ KRİTİK SAĞLIK UYARISI TESPİT EDİLDİ! '
                    'Lütfen derhal bir doktora başvurun.</div>',
                    unsafe_allow_html=True
                )

            # Rapor önizleme modal
            @st.dialog("📋 Tıbbi Analiz Raporu", width="large")
            def show_report_modal():
                sections = report.split('\n')
                current_section = []
                for line in sections:
                    if line.strip().startswith('#') or (line.strip() and line.strip()[0].isdigit() and '.' in line[:3]):
                        if current_section:
                            st.markdown(
                                f'<div class="report-section">{"<br>".join(current_section)}</div>',
                                unsafe_allow_html=True
                            )
                            current_section = []
                        st.markdown(f"#### {line.replace('#','').strip()}")
                    else:
                        if line.strip():
                            current_section.append(line)
                if current_section:
                    st.markdown(
                        f'<div class="report-section">{"<br>".join(current_section)}</div>',
                        unsafe_allow_html=True
                    )
                # TTS
                if st.button("🔊 Raporu Sesli Oku", key="modal_tts"):
                    with st.spinner("Ses oluşturuluyor..."):
                        tts_b = text_to_speech(report[:1200])
                    if tts_b:
                        st.markdown(get_audio_html(tts_b), unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔍 Raporu Önizle", use_container_width=True):
                    show_report_modal()
            with col_b:
                if st.button("📄 PDF İndir", use_container_width=True):
                    try:
                        p_path = generate_pdf_report(report)
                        with open(p_path, "rb") as f:
                            st.download_button(
                                "💾 Dosyayı Kaydet", f,
                                "PharmaAgent_Rapor.pdf", "application/pdf"
                            )
                    except Exception as e:
                        st.error(f"PDF Hatası: {e}")

            # Kısa rapor önizlemesi
            st.markdown('<div class="agent-card fade-in">', unsafe_allow_html=True)
            st.markdown("### 📋 Analiz Özeti")
            preview = report[:800] + ("..." if len(report) > 800 else "")
            st.markdown(
                f'<div style="background:rgba(0,0,0,0.2);padding:18px;border-radius:10px;'
                f'border:1px solid rgba(255,255,255,0.05);line-height:1.7;">{preview}</div>',
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # ── SOHBET ROBOTU ─────────────────────────────────────────────────
            st.markdown('<div class="agent-card fade-in">', unsafe_allow_html=True)
            st.markdown("### 💬 İlaç Asistanıyla Sohbet Et")
            st.caption("Rapor hakkında soru sorun — örn: *'Bu ilacı aç karnına alabilir miyim?'*")

            # Geçmiş mesajları göster
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

            # Yeni soru
            user_q = st.chat_input("Sorunuzu yazın...", key="chat_input_main")
            if user_q:
                with st.spinner("Düşünüyorum..."):
                    bot_ans = st.session_state.mas_v2.chat_with_report(
                        user_q, report, st.session_state.chat_history
                    )
                st.session_state.chat_history.append({"role": "user",    "content": user_q})
                st.session_state.chat_history.append({"role": "assistant", "content": bot_ans})

                # TTS seçeneği
                tts_col, _ = st.columns([1, 3])
                with tts_col:
                    if st.button("🔊 Cevabı Dinle", key=f"tts_{len(st.session_state.chat_history)}"):
                        tts_b = text_to_speech(bot_ans)
                        if tts_b:
                            st.markdown(get_audio_html(tts_b), unsafe_allow_html=True)
                st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.markdown(
            '<div class="agent-card fade-in" style="text-align:center;padding:60px 40px;">'
            '<div style="font-size:4rem;margin-bottom:1rem;">🩺</div>'
            '<h3 style="color:#60a5fa;">Akıllı Eczane\'ye Hoş Geldiniz</h3>'
            '<p style="color:#94a3b8;line-height:1.8;">'
            'Sol menüden isteğe bağlı sağlık profilinizi oluşturun.<br>'
            'Ardından bir <b>ilaç kutusu fotoğrafı</b> (opsiyonel: prospektüs kağıdını da) yükleyin.<br>'
            '7 yapay zeka ajanı saniyeler içinde kapsamlı bir tıbbi analiz sunar.</p>'
            '</div>',
            unsafe_allow_html=True
        )
