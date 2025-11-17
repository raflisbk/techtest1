# Explanation Document

**Nama:** Mohamad Rafli Agung Subekti

---

## Section 3: Reasoning & Explainability

### 1. Apa itu Overfitting dan Bagaimana Cara Mencegahnya?

**Overfitting** adalah kondisi dimana model machine learning terlalu menyesuaikan diri dengan data training sehingga memiliki performa sangat baik pada data training namun buruk pada data testing atau data baru. Model menjadi "menghafal" pola spesifik dalam data training termasuk noise, bukan mempelajari pola umum yang dapat digeneralisasi.

**Cara Mencegah Overfitting:**

1. **Regularization**

   - Menggunakan L1/L2 regularization untuk membatasi kompleksitas model
   - Menambahkan weight decay pada training parameters
   - Pada proyek ini: `weight_decay=0.01` diterapkan pada TrainingArguments
2. **Dropout**

   - Menonaktifkan sebagian neuron secara random saat training
   - Pada proyek ini: `lora_dropout=0.1` diterapkan pada LoRA configuration
3. **Early Stopping**

   - Menghentikan training ketika performa pada validation set mulai menurun
   - Pada proyek ini: `load_best_model_at_end=True` untuk menggunakan model terbaik
4. **Cross-Validation**

   - Memvalidasi model pada multiple splits data untuk memastikan generalisasi
   - Menggunakan stratified split untuk menjaga distribusi kelas
5. **Data Augmentation**

   - Memperbanyak variasi data training
   - Pada proyek ini: upsampling digunakan untuk balancing dataset
6. **Reduce Model Complexity**

   - Menggunakan model yang lebih sederhana atau parameter efficient fine-tuning
   - Pada proyek ini: LoRA (Low-Rank Adaptation) mengurangi trainable parameters menjadi ~1% dari total parameters
7. **More Training Data**

   - Mengumpulkan lebih banyak data training yang beragam
   - Pada proyek ini: scraping 10,000+ reviews dari Google Play Store

---

### 2. Jika Model Memprediksi "Positive" untuk Semua Review, Apa Implikasinya?

Jika model memprediksi **"Positive"** untuk semua review, hal ini mengindikasikan beberapa masalah serius:

**Implikasi Utama:**

1. **Class Imbalance Problem**

   - Dataset memiliki dominasi kelas "Positive" yang sangat tinggi
   - Model memilih strategi "safe" dengan selalu memprediksi kelas mayoritas
   - Accuracy mungkin tinggi (misleading), namun precision, recall, dan F1-score untuk kelas lainnya akan sangat rendah
2. **Model Tidak Belajar dengan Baik**

   - Model gagal menangkap pola pembeda antara kelas
   - Learning rate mungkin terlalu rendah atau training terlalu singkat
   - Feature representation tidak informatif untuk klasifikasi
3. **Bias dalam Data atau Preprocessing**

   - Proses cleaning text mungkin menghilangkan informasi penting
   - Tokenization atau feature extraction tidak optimal
   - Label assignment criteria tidak sesuai dengan konten review

**Solusi yang Diterapkan dalam Proyek:**

1. **Upsampling untuk Balancing**

   - Menyeimbangkan jumlah sampel di setiap kelas (Positive, Neutral, Negative)
   - Menggunakan stratified sampling untuk menjaga distribusi kelas
2. **Appropriate Evaluation Metrics**

   - Menggunakan weighted precision, recall, dan F1-score
   - Monitoring confusion matrix untuk setiap kelas
   - Tidak hanya mengandalkan accuracy sebagai metrik utama
3. **Class Weighting**

   - Pada Logistic Regression: `class_weight='balanced'`
   - Memberikan penalty lebih besar untuk kesalahan prediksi pada kelas minoritas
4. **Model Evaluation per Class**

   - Classification report menunjukkan performa setiap kelas
   - Confusion matrix untuk melihat pola misclassification

---

### 3. Bagaimana Meningkatkan Akurasi Model dengan Data yang Terbatas?

**Strategi Optimasi dengan Limited Data:**

#### A. Transfer Learning

- **Menggunakan Pre-trained Models**
  - Pada proyek ini: `cahya/distilbert-base-indonesian` yang sudah dilatih pada corpus bahasa Indonesia
  - Model sudah memahami struktur bahasa dan semantik dasar
  - Fine-tuning hanya perlu menyesuaikan untuk task spesifik

#### B. Parameter-Efficient Fine-Tuning (PEFT)

- **LoRA (Low-Rank Adaptation)**
  - Hanya melatih ~1% dari total parameters model
  - Mengurangi risiko overfitting pada dataset kecil
  - Computational efficient, memory efficient
  - Configuration: `r=16, lora_alpha=32, target_modules=["q_lin", "k_lin", "v_lin", "out_lin"]`

#### C. Data Augmentation

- **Text Augmentation Techniques:**

  - Synonym replacement untuk variasi vocabulary
  - Back-translation (Indonesia → English → Indonesia)
  - Random insertion/deletion/swap words
  - Paraphrasing menggunakan language models
- **Balancing dengan Resampling:**

  - Upsampling kelas minoritas untuk distribusi seimbang
  - Stratified sampling untuk mempertahankan distribusi

#### D. Feature Engineering

- **Advanced Text Preprocessing:**

  - Emoji handling dengan `emoji.demojize()`
  - URL, hashtag, dan mention removal
  - Repeated character normalization
  - Stopwords removal (optional, sesuai domain)
- **TF-IDF Optimization:**

  - `max_features=5000` untuk traditional ML models
  - `ngram_range=(1,2)` untuk capture phrase patterns
  - `min_df=2` untuk filter rare terms

#### E. Ensemble Methods

- **Combining Multiple Models:**
  - Voting classifier dari Logistic Regression, Naive Bayes, dan DistilBERT
  - Stacking dengan meta-learner
  - Weighted averaging berdasarkan model confidence

#### F. Hyperparameter Optimization

- **Systematic Tuning:**
  - Learning rate scheduling: `lr_scheduler_type='cosine'`
  - Optimal warmup steps: `warmup_ratio=0.1`
  - Batch size optimization: `batch_size=32`
  - Training epochs sesuai dataset size

#### G. Cross-Validation

- **K-Fold Cross-Validation:**
  - Memaksimalkan penggunaan limited data
  - Validasi robustness model pada multiple splits
  - Stratified K-Fold untuk maintain class distribution

#### H. Domain-Specific Approaches

- **Sentiment Lexicon:**
  - Menggunakan Indonesian sentiment dictionary
  - Feature combination: lexicon-based + ML-based
  - Rule-based preprocessing untuk domain healthcare

---

## Section 4: Deployment Thought Exercise

### Deployment Pipeline untuk 1,000 Reviews per Minute

Untuk menangani **1,000 reviews per menit** (~16.67 reviews/detik), saya akan merancang deployment pipeline berbasis **microservices architecture** dengan komponen berikut:

**(1) Load Balancer** dengan **NGINX** atau **AWS ALB** untuk mendistribusikan traffic secara merata ke multiple instances; **(2) API Gateway** menggunakan **FastAPI** dengan async endpoints untuk handling concurrent requests dengan efisien; **(3) Message Queue** seperti **Redis** atau **RabbitMQ** untuk buffering incoming reviews dan menghindari request loss saat traffic spike; **(4) Model Serving** menggunakan **TorchServe** atau **ONNX Runtime** dengan model optimization (quantization, pruning) untuk inference yang lebih cepat, dideploy pada **Kubernetes pods** dengan **horizontal auto-scaling** (min: 3 replicas, max: 10 replicas berdasarkan CPU/memory usage); **(5) Caching Layer** dengan **Redis** untuk menyimpan hasil prediksi reviews yang sama (deduplication) dan mengurangi redundant inference; **(6) Database** **PostgreSQL** untuk logging dan analytics, dengan **async writes** agar tidak blocking inference pipeline; **(7) Monitoring & Alerting** menggunakan **Prometheus + Grafana** untuk tracking latency, throughput, error rate, dan resource utilization dengan threshold-based alerts.

Pipeline ini akan menggunakan **batching strategy** (batch_size=32) untuk optimize GPU/CPU utilization, dengan **maximum latency target 200ms per request** untuk memastikan real-time experience. Deployment pada **cloud platform** (AWS/GCP/Azure) dengan **auto-scaling group** dan **CDN caching** untuk static assets akan menjamin scalability dan high availability dengan **99.9% uptime SLA**.

---

## Ringkasan Arsitektur Deployment

```
User Request (1000/min)
    ↓
Load Balancer (NGINX/ALB)
    ↓
API Gateway (FastAPI Async)
    ↓
Message Queue (Redis/RabbitMQ)
    ↓
Model Inference (TorchServe + K8s)
    ├── Pod 1 (DistilBERT + LoRA)
    ├── Pod 2 (DistilBERT + LoRA)
    └── Pod N (Auto-scaled)
    ↓
Caching Layer (Redis)
    ↓
Response + Logging (PostgreSQL)
    ↓
Monitoring (Prometheus + Grafana)
```

**Key Metrics:**

- **Throughput:** 1,000 reviews/minute = 16.67 req/sec
- **Target Latency:** < 200ms per request
- **Model Size:** ~200MB (DistilBERT + LoRA)
- **Inference Time:** ~50-80ms per review (CPU), ~10-20ms (GPU)
- **Batch Inference:** 32 reviews/batch untuk optimize throughput
