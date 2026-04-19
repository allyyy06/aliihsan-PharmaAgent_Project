# 💊 Akıllı Eczane — Yapay Zeka Destekli İlaç Analiz Platformu

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?logo=streamlit)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3_%26_4_Scout-green)
![Gemini](https://img.shields.io/badge/Google-Gemini_1.5-orange?logo=google)
![License](https://img.shields.io/badge/Lisans-MIT-lightgrey)

> 👨‍💻 ***Tasarlayan ve Geliştiren:** Ali İhsan ÇETİN*

[🇹🇷 Türkçe](#türkçe) | [🇬🇧 English](#english)

---

<a id="türkçe"></a>
## 🇹🇷 Türkçe

### 🚀 Proje Hakkında

**Akıllı Eczane**, herhangi bir ilaç kutusunun fotoğrafını yükleyerek saniyeler içinde kapsamlı bir tıbbi analiz sunan, yapay zeka destekli **Çok Ajanlı Sistem (Multi-Agent System)**'dir.

7 özelleşmiş yapay zeka ajanı koordineli çalışarak:
- 📷 İlaç kutusunu okur (Görsel Analiz — Gemini + Groq)
- 📚 Prospektüs veritabanını tarar (RAG — FAISS + HuggingFace)
- 🌐 Güncel FDA/TITCK uyarılarını canlı internetten çeker (DuckDuckGo)
- 🧬 **Sizi tanıyarak** kişisel risk değerlendirmesi yapar
- 💊 Alternatif ilaç önerir (Eczacı Ajanı)
- 🏢 Üretici firmayı denetler
- 📋 Tüm bunları tek bir **profesyonel PDF raporu** halinde sunar

---

### ✨ Özellikler (v2.0)

| # | Özellik | Açıklama |
|---|---------|----------|
| 1 | 👤 **Kişisel Sağlık Profili** | Ad, yaş, kronik hastalık, alerji girişi ile kişiselleştirilmiş analiz |
| 2 | 🔍 **Metin & 📷 Görsel Arama** | İlaç ismini doğrudan yazın veya görsel (kutu+prospektüs) yükleyerek analizi başlatın |
| 3 | 🌐 **Canlı Web Araması** | FDA/TITCK/EMA'dan anlık toplatma ve uyarı taraması |
| 4 | 🛡️ **Güvenlik Denetimi** | Yan etki, uyarı ve ölümcül risk tespiti (KIRMIZI ALARM) |
| 5 | 💊 **Eczacı Ajanı** | Risk durumunda alternatif ilaç önerileri |
| 6 | 🎯 **Risk Göstergesi** | Plotly ile 0–100 dinamik Gauge (kadran) grafiği |
| 7 | 💬 **Sohbet Robotu** | Rapor bağlamında Türkçe anlık Q&A |
| 8 | 🎙️ **Sesli Asistan** | Ana araç çubuğunda Groq Whisper STT + gTTS ile tam Türkçe ses desteği |
| 9 | 🔍 **Rapor Önizleme** | PDF indirmeden önce tam ekran modal görünüm |
| 10 | 📄 **PDF Raporu** | Tam analizi .PDF olarak kaydetme |
| 11 | 🎨 **Premium Kompakt UI** | Sadelik odaklı yatay araç çubuğu, Glassmorphism CSS, gradient animasyonlar |

---

### 🛠️ Teknoloji Yığını

| Katman | Teknoloji |
|--------|-----------|
| **Arayüz** | Streamlit + Custom CSS (Glassmorphism, Outfit/Inter font) |
| **Görsel AI** | Google Gemini 1.5 Flash → Groq Llama 4 Scout (Fallback) |
| **Metin AI** | Groq Llama 3.3 70B Versatile |
| **Hafıza (RAG)** | FAISS + LangChain + HuggingFace Embeddings |
| **Web Arama** | DuckDuckGo Search (API gerektirmez) |
| **STT** | Groq Whisper Large v3 Turbo |
| **TTS** | gTTS (Türkçe) |
| **PDF** | FPDF2 |
| **Grafikler** | Plotly |

---

### 📁 Proje Yapısı

```
AkilliEczane/
├── app.py              ← Ana uygulama (Streamlit UI — tüm özellikler)
├── agents.py           ← 7 yapay zeka ajanı (Vision, RAG, Web, Safety, Pharma, Corp, Chat)
├── tts_utils.py        ← Sesli asistan (Groq Whisper STT + gTTS TTS)
├── utils.py            ← RAG kurulumu + PDF oluşturucu
├── requirements.txt    ← Tüm bağımlılıklar
├── .gitignore          ← .env ve data/vector_db hariç tutulmuş
├── .streamlit/
│   └── config.toml     ← Streamlit Cloud tema ayarları
├── assets/
│   └── pharmacy_logo.png ← Eczane temalı sidebar logo (özel üretim)
└── data/
    ├── corpus/         ← PDF/TXT prospektüs dosyaları (buraya ekleyin)
    ├── vector_db/      ← FAISS indeks (otomatik oluşturulur, git'e dahil değil)
    └── reports/        ← Oluşturulan PDF raporlar
```

---

### ⚙️ Kurulum (Lokal)

```bash
# 1. Projeyi klonlayın
git clone https://github.com/allyyy06/aliihsan-PharmaAgent_Project.git
cd aliihsan-PharmaAgent_Project

# 2. Bağımlılıkları yükleyin
pip install -r requirements.txt

# 3. API anahtarlarını ayarlayın
# .env dosyası oluşturun:
echo "GOOGLE_API_KEY=your_key" > .env
echo "GROQ_API_KEY=your_key" >> .env

# 4. Çalıştırın
streamlit run app.py
```

---

### ☁️ Streamlit Cloud'a Deploy Etme

> [!IMPORTANT]
> **.env dosyası GitHub'a yüklenmez.** API anahtarlarınızı Streamlit Cloud'daki "Secrets" bölümüne gireceksiniz.

**Adım adım:**

1. **GitHub'a yükle:** `git push origin main`
2. **[share.streamlit.io](https://share.streamlit.io)** adresine gidin
3. **"New app"** → Repo seçin → Branch: `main` → File: `app.py`
4. **"Advanced settings > Secrets"** bölümüne şunu yapıştırın:
   ```toml
   GOOGLE_API_KEY = "your_google_api_key"
   GROQ_API_KEY   = "your_groq_api_key"
   ```
5. **"Deploy"** butonuna basın → Birkaç dakika içinde canlıya alınır ✅

---

### 🔑 API Anahtarları Nereden Alınır?

| Servis | Link | Ücretsiz Kota |
|--------|------|---------------|
| Google Gemini | [aistudio.google.com](https://aistudio.google.com) | ✅ Var |
| Groq (Llama + Whisper) | [console.groq.com](https://console.groq.com) | ✅ Generöz kota |

---

### 📖 Kullanım Kılavuzu

1. **API Ayarları** → Proje ana dizinindeki `.env` dosyasını açıp kendi Google ve Groq API anahtarlarınızı yapıştırın. Sistem anahtarları arayüzden gizleyerek sadece arka plandan okuyacaktır.
2. **Kişisel Sağlık Profili** → İsteğe bağlı: sol panelden sağlık bilgilerinizi doldurun
3. **Analizi Başlatın** → Üst kısımdaki araç çubuğundan doğrudan ilaç adını yazın veya ilaç kutusu fotoğrafı yükleyin ve "Analizi Başlat" butonuna tıklayın.
4. **Analizi İzleyin** → 7 ajan sırayla çalışır, sonuçlar canlı gösterilir
5. **Raporu İnceleyin** → "Raporu Önizle" veya "PDF İndir"
6. **Sohbet Edin** → Rapor hakkında soru sorun veya sesli asistanı kullanın

---

> ⚠️ *Bu platform sağlık alanında **bilinçlendirme** amacıyla geliştirilmiştir. Profesyonel tıbbi tavsiyenin yerini tutmaz. Her zaman doktorunuza danışın.*

*Tasarlayan ve Geliştiren: **Ali İhsan ÇETİN***

---

<a id="english"></a>
## 🇬🇧 English

### About

**Akıllı Eczane** (*Smart Pharmacy*) is an advanced AI-powered **Multi-Agent System (MAS)** that delivers a comprehensive pharmaceutical analysis within seconds — simply by uploading a photo of any medicine box.

### Key Features (v2.0)

- **Personalized Health Profile** — Age, chronic diseases, allergies, current medications → personalized risk assessment
- **Smart Text & Image Search** — Search precisely by directly typing the drug name or uploading images (Box + leaflet).
- **Live Web Search** — Real-time FDA/EMA recall lookup via DuckDuckGo (no API key needed)
- **Safety Auditor** — Detects side effects, contraindications, life-threatening risks (RED ALARM)
- **Pharmacist Agent** — Suggests safer alternatives when risks are detected
- **Risk Gauge** — Dynamic Plotly gauge chart (0–100 risk score)
- **Chat Interface** — Natural language Q&A about the report in Turkish
- **Voice Assistant** — Integrated prominently in the top toolbar with Groq Whisper STT + gTTS TTS, fully Turkish
- **Report Preview Modal** — Full-screen clean preview before PDF download
- **PDF Export** — Complete analysis downloadable as PDF
- **Premium Compact UI** — Simple, horizontal toolbar, custom CSS, gradient animations, and glassmorphism.

### Quick Start

```bash
pip install -r requirements.txt
# Create .env with GOOGLE_API_KEY and GROQ_API_KEY
streamlit run app.py
```

### Streamlit Cloud Deploy

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select repo/branch/file
3. Add secrets in **Settings > Secrets**:
   ```toml
   GOOGLE_API_KEY = "..."
   GROQ_API_KEY   = "..."
   ```
4. Click **Deploy** ✅

---

*Designed & developed by **Ali İhsan ÇETİN** for educational purposes. Not a substitute for professional medical advice.*
