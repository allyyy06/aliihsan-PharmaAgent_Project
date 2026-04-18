import os
import json
import base64
import io
from groq import Groq
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN MAS CLASS
# ─────────────────────────────────────────────────────────────────────────────
class PharmaGuardMAS:
    def __init__(self, google_api_key, groq_api_key):
        self.google_api_key = google_api_key
        self.groq_api_key = groq_api_key
        self.vision_engine = "Gemini"
        self.last_used_model = "Baslatiliyor..."

        # 1. Initialize Google Gemini
        try:
            genai.configure(api_key=google_api_key, transport='rest')
            self.selected_vision = 'gemini-flash-latest'
            self.selected_pro = 'gemini-pro-latest'
            self.vision_model = genai.GenerativeModel(self.selected_vision)
        except Exception as e:
            print(f"DEBUG: Google Config Hatası: {e}")
            self.selected_vision = 'gemini-flash-latest'
            self.selected_pro = 'gemini-pro-latest'
            self.vision_model = None

        # 2. Initialize Groq
        try:
            self.groq_client = Groq(api_key=groq_api_key)
        except Exception as e:
            print(f"DEBUG: Groq Init Hatası: {e}")
            self.groq_client = None

    def _call_llm(self, system_prompt, user_prompt, messages=None):
        """
        KURSUN GECIRMEZ LLM CAGIRICI
        Sıralama: Groq Llama-70B -> Groq Llama-8B -> Google Gemini Pro -> Hata Mesajı
        """
        def build_msgs(sys, usr):
            m = []
            if sys: m.append({"role": "system", "content": sys})
            m.append({"role": "user", "content": usr})
            return m

        # --- AŞAMA 1: GROQ LLAMA-70B ---
        if self.groq_client:
            try:
                msgs = messages or build_msgs(system_prompt, user_prompt)
                chat_completion = self.groq_client.chat.completions.create(
                    messages=msgs,
                    model="llama-3.3-70b-versatile",
                    timeout=20
                )
                self.last_used_model = "Llama-70B (Groq)"
                content = chat_completion.choices[0].message.content
                if content: return content
            except Exception as e:
                print(f"DEBUG: Groq 70B Basarisiz (Kota/Hata): {e}")

        # --- AŞAMA 2: GROQ LLAMA-8B (Hızlı/Yedek) ---
        if self.groq_client:
            try:
                msgs = messages or build_msgs(system_prompt, user_prompt)
                chat_completion = self.groq_client.chat.completions.create(
                    messages=msgs,
                    model="llama-3.1-8b-instant",
                    timeout=15
                )
                self.last_used_model = "Llama-8B (Groq - Yedek)"
                content = chat_completion.choices[0].message.content
                if content: return content
            except Exception as e:
                print(f"DEBUG: Groq 8B Basarisiz: {e}")

        # --- AŞAMA 3: GOOGLE GEMINI PRO (Tam Yedek) ---
        try:
            model_to_use = genai.GenerativeModel(
                model_name=self.selected_pro,
                system_instruction=system_prompt if (system_prompt and not messages) else None
            )
            if messages:
                full_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
                response = model_to_use.generate_content(full_prompt)
            else:
                response = model_to_use.generate_content(user_prompt)
            
            self.last_used_model = "Gemini Pro (Google - Yedek)"
            return response.text
        except Exception as e:
            msg = f"Tüm modeller (Groq & Gemini) başarısız oldu. Lütfen internet bağlantınızı veya API anahtarlarınızı kontrol edin. Detay: {str(e)}"
            self.last_used_model = "Kritik Hata"
            return msg

    # ─── AGENTS ───────────────────────────────────────────────────────────────

    def vision_scanner(self, image_bytes_list: list) -> dict:
        """İlaç kutusu/prospektüs görsellerini analiz eder."""
        prompt = (
            "Bu ilaç kutusu/prospektüs fotoğraf(lar)ını analiz et. "
            "JSON Formatında: Ilac_Adi, Etken_Madde, Dozaj_mg, Form, Uretici_Firma, Barkod. "
            "SADECE ham JSON döndür."
        )
        errors = []

        # 1. Gemini Vision
        if self.vision_model:
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    parts = [prompt]
                    for img_bytes in image_bytes_list:
                        parts.append({"mime_type": "image/jpeg", "data": img_bytes})
                    response = self.vision_model.generate_content(parts)
                    if response and response.text:
                        json_str = response.text.replace('```json', '').replace('```', '').strip()
                        self.vision_engine = "Gemini Vision"
                        return json.loads(json_str)
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg and attempt < max_retries - 1:
                        print(f"DEBUG: Gemini Kotası Dolu, 10sn bekleniyor (Deneme {attempt+1})...")
                        import time
                        time.sleep(10)
                        continue
                    
                    err_detail = f"Gemini Hatası: {err_msg}"
                    print(f"DEBUG: {err_detail}")
                    errors.append(err_detail)
                    break # Başka hataysa döngüden çık
        else:
            errors.append("Gemini model başlatılamadı (Anahtar eksik olabilir).")

        # 2. Groq Vision Fallback
        if self.groq_client:
            try:
                base64_image = base64.b64encode(image_bytes_list[0]).decode('utf-8')
                completion = self.groq_client.chat.completions.create(
                    model="llama-3.2-11b-vision-instant",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }],
                    response_format={"type": "json_object"}
                )
                self.vision_engine = "Groq Vision"
                return json.loads(completion.choices[0].message.content)
            except Exception as e:
                err_msg = f"Groq Vision Hatası: {str(e)}"
                print(f"DEBUG: {err_msg}")
                errors.append(err_msg)
        else:
            errors.append("Groq client başlatılamadı (Anahtar eksik olabilir).")

        final_error = " | ".join(errors)
        return {"error": f"Görsel analiz motorları başarısız oldu: {final_error}", "Ilac_Adi": "Bilinmeyen"}

    def web_search(self, drug_name="Bilinmeyen"):
        """Canlı internet uyarıları."""
        results_text = "İnternet araması yapılamadı."
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                search_query = f"{drug_name} yan etkileri TITCK FDA toplatılma uyarısı"
                results = list(ddgs.text(search_query, max_results=4))
                results_text = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        except: pass

        sys_p = "Sen kıdemli bir Sağlık Güvenlik Analistisin. İnternet arama sonuçlarını özetle. KESİNLİKLE SADECE TÜRKÇE DİLİNDE YANIT VER."
        user_p = f"İlaç: {drug_name}\n\nArama Sonuçları:\n{results_text}"
        return self._call_llm(sys_p, user_p)

    def rag_specialist(self, drug_name, vector_db):
        """Prospektüs veritabanı sorgusu."""
        context = "Yerel veritabanında bilgi bulunamadı."
        if vector_db:
            try:
                results = vector_db.similarity_search(drug_name, k=3)
                context = "\n".join([doc.page_content for doc in results])
            except: pass
            
        sys_p = "Sen bir Tıbbi Veri Uzmanısın. Verilen prospektüs verilerini özetle. KESİNLİKLE SADECE TÜRKÇE DİLİNDE YANIT VER."
        user_p = f"İlaç: {drug_name}\n\nVeriler:\n{context}"
        return self._call_llm(sys_p, user_p)

    def safety_auditor(self, context, user_profile=None):
        """Kişisel sağlık profili analizi."""
        profile_text = "Profil belirtilmedi."
        if user_profile:
            profile_text = f"İsim: {user_profile.get('name')}, Yaş: {user_profile.get('age')}, Hastalıklar: {user_profile.get('diseases')}, Alerjiler: {user_profile.get('allergies')}"

        sys_p = "Sen Bağımsız bir İlaç Güvenlik Denetçisisin. Ölümcül riskleri tespit et. KESİNLİKLE SADECE TÜRKÇE DİLİNDE YANIT VER."
        user_p = f"PROFİL: {profile_text}\n\nVERİ: {context}\n\nKişisel riskleri ve 'KIRMIZI ALARM' durumlarını belirt."
        return self._call_llm(sys_p, user_p)

    def pharmacist_agent(self, safety_data, vision_data, user_profile=None):
        """Alternatif ilaç önerici."""
        drug_name = vision_data.get("Ilac_Adi", "bilinmeyen")
        sys_p = "Sen uzman bir Eczacısın. Güvenlik risklerini değerlendirip alternatif çözümler önerirsin. KESİNLİKLE SADECE TÜRKÇE DİLİNDE YANIT VER."
        user_p = f"İlaç: {drug_name}\nGüvenlik Raporu: {safety_data[:600]}\nRisk varsa 2 alternatif öner."
        return self._call_llm(sys_p, user_p)

    def corporate_analyst(self, drug_details):
        """Üretici firma analizi."""
        sys_p = "Sen bir Kurumsal Risk Analistisin. Üretici firma güvenilirliğini analiz et. KESİNLİKLE SADECE TÜRKÇE DİLİNDE YANIT VER."
        user_p = f"Veri: {drug_details}. Üretici hakkında kısa rapor yaz."
        return self._call_llm(sys_p, user_p)

    def extract_risk_score(self, safety_data):
        """0-100 arası risk skoru."""
        sys_p = "Sadece 0-100 arası bir sayı döndür."
        user_p = f"Bu raporun risk skoru nedir? {safety_data[:500]}"
        res = self._call_llm(sys_p, user_p)
        try:
            import re
            nums = re.findall(r'\d+', res)
            return int(nums[0]) if nums else 30
        except: return 30

    def synthesize_report(self, vision_data, rag_data, safety_data, corp_data, web_data, pharmacist_data, user_profile=None):
        """Nihai rapor oluşturma."""
        sys_p = "Sen Akıllı Eczane'nin Baş Sentez Uzmanısın. Profesyonel bir tıbbi rapor hazırla. KESİNLİKLE SADECE TÜRKÇE DİLİNDE YANIT VER."
        user_p = f"Tüm veriler: {vision_data}, {rag_data}, {safety_data}, {corp_data}, {web_data}, {pharmacist_data}"
        return self._call_llm(sys_p, user_p)

    def chat_with_report(self, question, report_context, history):
        """Rapor bağlamında asistan sohbeti."""
        system = f"Sen Akıllı Eczane asistanısın. KESİNLİKLE SADECE YALIN TÜRKÇE KULLAN. Çince veya yabancı karakterler KULLANMA. Rapor: {report_context[:1500]}"
        messages = [{"role": "system", "content": system}]
        for h in history[-4:]: messages.append(h)
        messages.append({"role": "user", "content": question})
        return self._call_llm(system, question, messages=messages)
