# 🚀 LLM Fundamentals & Prompting

Bu proje; Büyük Dil Modelleri (LLM) ile uygulama geliştirirken kullanılan temel **Prompt Mühendisliği** tekniklerini, **API Parametrelerinin** yanıtlar üzerindeki etkilerini ve RAG sistemlerinin kalbi olan **Context Injection & Grounding (Bağlam Enjeksiyonu ve Doğrulama)** süreçlerini deneysel olarak incelemek amacıyla hazırlanmıştır.

Çalışmada performans ve erişim kolaylığı sebebiyle **Groq API** altyapısı üzerinden **Llama 3.3 70B (`llama-3.3-70b-versatile`)** modeli kullanılmış ve tüm senaryolar adım adım Jupyter Notebook ortamında test edilmiştir.

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler

Proje kapsamında yararlanılan temel araçlar ve kullanım amaçları şunlardır:

* **Python 3.10+:** Tüm test senaryolarının ve API çağrılarının kurgulandığı temel programlama dili.
* **Jupyter Notebook:** Deneylerin adım adım çalıştırıldığı, girdi/çıktı analizlerinin ve parametre değişimlerinin interaktif olarak gözlemlendiği ortam.
* **Groq SDK (`groq`) & OpenAI SDK (`openai`):** Llama 3.3 modeline yüksek hızlı (LPUs - Language Processing Units) çıkarım (inference) çağrıları yapmamızı sağlayan istemci kütüphaneleri.
* **Llama 3.3 70B (`llama-3.3-70b-versatile`):** Meta tarafından geliştirilen, açık ağırlıklı (open-weights) ve yüksek akıl yürütme kapasitesine sahip dil modeli.
* **Python-Dotenv (`python-dotenv`):** API anahtarlarının koda gömülmeden, ortam değişkenleri veya `.env` yapılandırması üzerinden güvenli bir şekilde yönetilmesini sağlayan araç.

---

## 📌 Notebook İçerisindeki Deneyler ve Örnekler (`llm_prompting_and_parameters.ipynb`)

Notebook boyunca uygulanan ve çıktıları analiz edilen tüm pratik örnekler:

### 1. Mesaj Hiyerarşisi ve Roller (System / User / Assistant)
* Model davranışını ve karakterini belirleyen `system` mesajlarının `user` istemleri üzerindeki otoritesi test edildi.

### 2. Yapılandırılmış Çıktı ve Format Sabitleme (JSON Output)
* Modelden serbest metin yerine, yazılım backend/API sistemlerinin doğrudan ayrıştırabileceği (parse edebileceği) geçerli **JSON** formatında veri üretmesi sağlandı.

### 3. Örnekle Öğretme (Few-Shot Prompting)
* Modele ne yapacağı sadece açıklanmadı; girdi-çıktı örnekleri (`Few-shot`) verilerek karmaşık veya formata duyarlı görevlerde tutarlılık artırıldı.

### 4. Akıl Yürütme ve Adım Adım Düşünme (Chain-of-Thought - CoT)
* Zorlu problemlerde modele *"Adım adım düşün"* talimatı verilerek ara adımları kağıda dökmesi sağlandı ve mantıksal hatalar minimize edildi.

### 5. API Parametre Deneyleri (Parameter Tuning)
* **Temperature Testi:** $T = 0.0$ (deterministik ve odaklı) ile $T = 1.0$ (yaratıcı ve çeşitli) değerlerinin yanıt dağılımına etkisi gözlemlendi.
* **Max Tokens & Finish Reason:** Token sınırı düşürülerek yanıtın yarıda kesilmesi sağlandı; API'den dönen `finish_reason: length` bayrağının backend tarafında nasıl yakalanacağı deneyimlendi.

### 6. Context Injection & Grounding (RAG Temeli)
* **Temel Bağlam Testi:** Şirket izin politikası metni bağlam olarak verilerek modelin cevabı cımbızla çekmesi sağlandı.
* **Hallucination Control (Fall-back Strategy):** Metinde **bulunmayan** bir soru (*"Doğum izni kaç gün?"*) soruldu. Prompta koyulan *"Cevap yoksa 'Bu bilgi verilen belgede yok' de"* kaçış kuralının halüsinasyonu (uydurmayı) nasıl engellediği kanıtlandı.
* **Kaçış Satırı Kaldırma Deneyi:** B Planı talimatı silindiğinde modelin net cevap vermek yerine durumu kıvırmaya ve açıklamayı uzatmaya çalıştığı gözlemlendi.
* **Poisoned Context (Kasten Yanlış Bilgi) Testi:** Bağlama kasten *"Yıllık izin 45 gündür"* bilgisi eklendi. Model genel kültür verisini unutup önündeki 45 gün bilgisini verdi. Böylece **In-Context Learning > Pre-trained Knowledge** ilkesi doğrulandı.
* **Kişisel Not / Staj Bağlam Testi:** Gerçekçi staj/proje notlarından oluşan bir metin verilerek modelin çoklu sorulara doğru yanıt üretme hassasiyeti doğrulandı.

---

## ⚡ Kurulum ve Çalıştırma

Projeyi kendi ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

### 1. Repoyu Klonlayın
```bash
git clone [https://github.com/ElifBerra/llm-fundamentals-and-prompting.git](https://github.com/ElifBerra/llm-fundamentals-and-prompting.git)
cd llm-fundamentals-and-prompting

```

### 2. Gerekli Kütüphaneleri Yükleyin

```bash
pip install groq openai python-dotenv jupyter

```

### 3. 🔑 API Anahtarınızı Tanımlayın (Önemli)

Bu projeyi çalıştırabilmek için kendi **Groq API Key** değerinize ihtiyacınız vardır. *(Ücretsiz API key almak için: [console.groq.com](https://console.groq.com/))*

API anahtarınızı iki farklı şekilde tanımlayabilirsiniz:

* **Yöntem A (Önerilen - `.env` Dosyası):**
Proje dizininde `.env` adında bir dosya oluşturup içine kendi anahtarınızı ekleyin:
```env
GROQ_API_KEY=gsk_sizin_gercek_groq_api_keyiniz

```


* **Yöntem B (Notebook İçinde):**
Notebook'un ilgili hücresinde bulunan `groq_key` değişkenine kendi anahtarınızı yazabilirsiniz:
```python
groq_key = "gsk_sizin_gercek_groq_api_keyiniz"

```


*(⚠️ **Uyarı:** Kendi API anahtarınızı içeren Notebook dosyasını GitHub gibi açık kaynaklı platformlara yüklemeyiniz!)*

### 4. Notebook'u Başlatın

```bash
jupyter notebook llm_prompting_and_parameters.ipynb

```

---

## 💡 Temel Mühendislik Çıkarımları

1. **If you don't give the LLM a way to say 'I don't know', it will guess:** Modele "Bilmiyorum" deme seçeneği (Fall-back) sunulmazsa model halüsinasyon görmeye meyillidir.
2. **Deterministic Output for Systems:** Backend sistemlerine entegre edilecek kodlarda $T = 0.0$ ve `Structured Output` (JSON) kullanımı şarttır.
3. **Data Quality is Everything:** RAG sistemlerinde veritabanına giren veri neyse model onu doğru kabul eder (Garbage in, Garbage out).

---

## 📜 Lisans

Bu proje eğitim ve araştırma amacıyla geliştirilmiştir. [MIT License](https://www.google.com/search?q=LICENSE) altında açık kaynak olarak sunulmuştur.
