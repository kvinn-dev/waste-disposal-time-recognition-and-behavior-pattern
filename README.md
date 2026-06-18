# Waste Disposal Time Pattern Recognition
> Computer Vision · Data Fusion · User Behavior Clustering · Time Series Forecasting

---

## Overview

Waste Disposal Time Pattern Recognition adalah proyek Machine Learning yang menggabungkan Computer Vision, Data Fusion, Clustering, dan Time Series Forecasting untuk menganalisis pola pembuangan sampah berdasarkan jenis sampah, waktu, lokasi, dan perilaku pengguna.

Proyek ini dikembangkan sebagai implementasi end-to-end pipeline Machine Learning yang mencakup:

* Deteksi objek sampah menggunakan YOLOv8
* Integrasi data visual dan temporal (Data Fusion)
* Analisis eksploratif dataset (EDA)
* Segmentasi perilaku pengguna menggunakan Clustering
* Peramalan volume sampah menggunakan Prophet dan LSTM
* Penyusunan insight untuk mendukung Smart Waste Management

---

## Research Objectives

Penelitian ini bertujuan untuk:

* Mendeteksi objek sampah dari gambar menggunakan Computer Vision.
* Mengintegrasikan data visual dan temporal menjadi dataset yang lebih kaya.
* Mengidentifikasi pola waktu pembuangan sampah.
* Menentukan jam puncak dan periode aktivitas tertinggi.
* Mengelompokkan perilaku pengguna berdasarkan kebiasaan membuang sampah.
* Memprediksi volume sampah di masa depan menggunakan pendekatan statistik dan Deep Learning.

---

## System Architecture

```text
┌─────────────────────┐
│ Image Dataset       │
│ (Roboflow Universe) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ YOLOv8 Detection    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Personal Image      │
│ Inference           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Data Fusion         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Feature Engineering │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ EDA & Analytics     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Clustering          │
│ K-Means & DBSCAN    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Forecasting         │
│ Prophet & LSTM      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Final Insights      │
└─────────────────────┘
```

---

# Datasets

## 1. Image Dataset

Dataset gambar diperoleh dari Roboflow Universe dan digunakan untuk melatih model YOLOv8 dalam mendeteksi berbagai jenis sampah.

### Dataset Information

| Attribute         | Value             |
| ----------------- | ----------------- |
| Source            | Roboflow Universe |
| Total Images      | 2,060             |
| Annotation Format | YOLOv8            |
| Classes           | 4                 |
| Task              | Object Detection  |

### Dataset Preview

![Dataset Preview](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/Screenshot%202026-06-18%20at%2011.49.10.png)

### Dataset Source

```text
https://universe.roboflow.com/waste-classification-ftcyk/multiple-waste-dataset-2
```

---

## Classes

| Class ID | Class Name    |
| -------- | ------------- |
| 0        | Biodegradable |
| 1        | Recyclable    |
| 2        | Residual      |
| 3        | Special       |

### Sample Images

![Sample Image 1](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/017_jpg.rf.d7f2673bf95b3d66e69d5dcd59e93935.jpg)

![Sample Image 2](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/081_jpg.rf.0fc204de560558da4316f1b4f664a82e.jpg)

![Sample Image 3](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/132_jpg.rf.e5020457b8dec39795e8fc9eedd1316f.jpg)

---

## 2. Temporal Dataset

Dataset temporal diperoleh dari Mendeley Data dan digunakan untuk membangun pola waktu pembuangan sampah.

### Dataset Information

| Attribute   | Value          |
| ----------- | -------------- |
| Source      | Mendeley Data  |
| Records     | 8,760          |
| Period      | Full Year 2025 |
| Granularity | Hourly         |

### Dataset Source

```text
https://data.mendeley.com/datasets/57869zvy6y/1
```

### Dataset Structure

| Column           | Description    |
| ---------------- | -------------- |
| Hour             | Jam pembuangan |
| Waste_kg         | Berat sampah   |
| System_Activated | Status sistem  |

### Dataset Preview

![CSV Dataset Preview](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/Screenshot%202026-06-18%20at%2011.56.53.png)

---

## 3. Personal Images

Dataset tambahan yang dikumpulkan secara mandiri untuk menguji kemampuan generalisasi model pada kondisi nyata.

### Characteristics

* 10 gambar sampah dunia nyata
* Berisi berbagai kombinasi sampah
* Digunakan untuk inferensi dan validasi model

### Examples

![Personal Image 1](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/001.jpg)

![Personal Image 2](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/006.jpg)

---

# YOLOv8 Object Detection

## Training Configuration

| Parameter  | Value     |
| ---------- | --------- |
| Model      | YOLOv8n   |
| Epoch      | 50        |
| Patience   | 10        |
| Image Size | 640 × 640 |
| Batch Size | 16        |
| Optimizer  | Adam      |
| GPU        | Tesla T4  |

---

## Dataset Split

| Split      | Images |
| ---------- | ------ |
| Train      | 1,902  |
| Validation | 80     |
| Test       | 78     |
| Total      | 2,060  |

---

## Training Yolov8 Results

![Training Result](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/Screenshot%202026-06-18%20at%2012.02.17.png)

---

## Validation Results

| Metric    | Value |
| --------- | ----- |
| Precision | 0.76  |
| Recall    | 0.542 |
| mAP50     | 0.644 |
| mAP50-95  | 0.488 |

---

## Per-Class Performance

| Class         | Precision | Recall | mAP50 | mAP50-95 |
| ------------- | --------- | ------ | ----- | -------- |
| Biodegradable | 0.752     | 0.517  | 0.606 | 0.426    |
| Recyclable    | 0.752     | 0.55   | 0.699 | 0.577    |
| Residual      | 0.831     | 0.6    | 0.741 | 0.588    |
| Special       | 0.706     | 0.5    | 0.53  | 0.361    |

---

## Detection Examples

### Before Detection

![Original Image](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/006.jpg)

### After Detection

![Detection Result](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/Screenshot%202026-06-18%20at%2012.07.07.png)

---

# Data Fusion

Data Fusion mengintegrasikan:

1. Hasil deteksi YOLOv8
2. Dataset temporal Mendeley
3. Metadata tambahan
4. Rekayasa fitur perilaku

### Data Fusion Pipeline

![Fusion Pipeline](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/Screenshot%202026-06-18%20at%2012.08.55.png)

---

## Sources

### YOLO Features

* detected_labels
* detected_count
* total_objects
* image_path

### Temporal Features

* timestamp
* waste distribution
* peak-hour statistics

### Metadata Features

* user_id
* kecamatan
* kelurahan
* tps_code
* seasonal_factor

---

# Final Dataset

## Dataset Summary

| Attribute | Value          |
| --------- | -------------- |
| Records   | 1,000+         |
| Features  | 20             |
| Source    | Fusion Dataset |

---

## Feature Categories

| Category      | Features                       |
| ------------- | ------------------------------ |
| Temporal      | hour, day_of_week, month       |
| Location      | kecamatan, kelurahan           |
| Detection     | detected_labels, total_objects |
| Behavioral    | is_peak_hour, is_weekend       |
| Environmental | seasonal_factor                |
| Weight        | weight_kg                      |

---

## Final Dataset Preview

![Final Dataset](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/Screenshot%202026-06-18%20at%2012.10.31.png)

---

# Exploratory Data Analysis

## Waste Type Distribution

![Waste Distribution](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/Screenshot%202026-06-18%20at%2012.11.21.png)

### Insight

Residual dan Recyclable merupakan kategori sampah yang paling dominan dalam dataset.

---

## Weight Distribution

![Weight Distribution](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/Screenshot%202026-06-18%20at%2012.12.07.png)

### Insight

Distribusi berat bersifat right-skewed dengan sebagian besar aktivitas memiliki berat relatif rendah.

---

## Hourly Distribution

![Hourly Distribution](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/Screenshot%202026-06-18%20at%2012.12.47.png)

---

## Weekly Distribution

![Weekly Distribution](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/Screenshot%202026-06-18%20at%2012.13.28.png)

---

## Monthly Distribution

![Monthly Distribution](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/Screenshot%202026-06-18%20at%2012.14.00.png)

---

# Temporal Analysis

## Peak Disposal Hours

| Period       | Time        |
| ------------ | ----------- |
| Morning Peak | 07:00–09:00 |
| Midday Peak  | 12:00–13:00 |
| Evening Peak | 18:00–19:00 |

### Heatmap

![Peak Hour Heatmap](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/Screenshot%202026-06-18%20at%2012.15.01.png)

### Findings

* Aktivitas tertinggi terjadi pada pagi dan malam hari.
* Pola pembuangan meningkat pada akhir pekan.
* Musim hujan meningkatkan volume sampah sebesar 10–20%.

---

# Location Analysis

## District Distribution

![District Distribution](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/Screenshot%202026-06-18%20at%2012.15.48.png)

Dataset mencakup 12 kecamatan Kota Bekasi.

---

# User Behavior Analysis

## Behavioral Features

* Average Disposal Hour
* Disposal Frequency
* Average Weight
* Weekend Ratio
* Peak Hour Ratio
* Consistency Score

### Visualization

![Behavior Analysis](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/usbehavior.png)

---

# K-Means Clustering

## Elbow Method

![Elbow Method](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/Screenshot%202026-06-18%20at%2012.17.46.png)

---

## Cluster Visualization

![KMeans Cluster](https://hdaphuncqwjfxysgmcew.supabase.co/storage/v1/object/public/machine-learning/Screenshot%202026-06-18%20at%2012.18.22.png)

---

## Results

| Cluster   | Description      |
| --------- | ---------------- |
| Cluster 1 | Frequent Users   |
| Cluster 2 | Regular Users    |
| Cluster 3 | Occasional Users |

### Silhouette Score

```text
0.1632
```

---

# DBSCAN Clustering

## Visualization

![DBSCAN](YOUR_SUPABASE_URL/clustering/dbscan_clusters.png)

### Characteristics

* Detects outliers
* Handles arbitrary cluster shapes
* Does not require predefined cluster count

---

# 📉 Prophet Forecasting

## Forecast Result

![Prophet Forecast](YOUR_SUPABASE_URL/forecasting/prophet_forecast.png)

---

## Forecast Components

![Prophet Components](YOUR_SUPABASE_URL/forecasting/prophet_components.png)

---

## Evaluation Metrics

| Metric | Value           |
| ------ | --------------- |
| MAE    | 5.245           |
| RMSE   | 6.910           |
| MAPE   | 39155428760.35% |

---

# 🧠 LSTM Forecasting

## Model Architecture

```text
Input Sequence (14 Steps)
        ↓
LSTM (64 Units)
        ↓
Dropout (0.2)
        ↓
LSTM (32 Units)
        ↓
Dropout (0.2)
        ↓
Dense Output Layer
```

---

## Training Loss

![LSTM Loss](YOUR_SUPABASE_URL/lstm/loss_curve.png)

---

## Prediction Result

![LSTM Prediction](YOUR_SUPABASE_URL/lstm/prediction_result.png)

---

## Evaluation Metrics

| Metric | Value |
| ------ | ----- |
| MAE    | 4.655 |
| RMSE   | 5.524 |

---

# 📊 Model Comparison

| Model   | Task               | Main Metric     | Result    |
| ------- | ------------------ | --------------- | --------- |
| YOLOv8  | Object Detection   | mAP50           | 0.634     |
| K-Means | User Segmentation  | Silhouette      | 0.1632    |
| DBSCAN  | Density Clustering | Noise Detection | Supported |
| Prophet | Forecasting        | RMSE            | 6.910     |
| LSTM    | Forecasting        | RMSE            | 5.524     |

---

# 🏆 Key Findings

✅ YOLOv8 berhasil mendeteksi empat kategori sampah.

✅ Model mampu melakukan generalisasi pada gambar dunia nyata.

✅ Data Fusion menghasilkan dataset kaya dengan 20 fitur.

✅ Pola waktu pembuangan memiliki tiga periode puncak utama:

* Pagi
* Siang
* Malam

✅ Aktivitas pembuangan meningkat pada akhir pekan.

✅ K-Means dan DBSCAN berhasil mengidentifikasi pola perilaku pengguna.

✅ LSTM memberikan performa forecasting terbaik dibanding Prophet pada eksperimen ini.

---

# 🚀 Future Work

* Real-Time IoT Integration
* Smart Bin Monitoring
* Live Waste Forecast Dashboard
* Edge AI Deployment
* Smart City Integration
* Mobile Monitoring Application

---

# 👨‍💻 Authors

### Kelompok 9

| Name                   | Student ID  |
| ---------------------- | ----------- |
| Anggiant Dwi Raka      | 20240801120 |
| Kevin Yulian Pamungkas | 20240801273 |
| Ramadhan Linggar Karan | 20220801013 |

---

## 📄 License

This project was developed for academic and educational purposes as part of the Machine Learning course project.

Machine Learning • 2026
