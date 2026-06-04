# Eksperimen_SML_Suryani_apc367d6x0436

## Deskripsi
Repository ini berisi eksperimen preprocessing dan pembangunan model machine learning untuk dataset Heart Disease UCI menggunakan Python dan scikit-learn.

## Struktur Repository
Eksperimen_SML_Suryani_apc367d6x0436/
├── .github/workflows/     # GitHub Actions workflow
├── .workflow/             # Workflow CI (ketentuan submission)
├── preprocessing/         # File preprocessing
│   ├── Eksperimen_SML_Suryani_apc367d6x0436.ipynb
│   ├── automate_Suryani.py
│   ├── heart_preprocessing_train.csv
│   └── heart_preprocessing_test.csv
└── heart_disease_uci.csv  # Dataset raw

## Dataset
- **Sumber**: [Heart Disease UCI - Kaggle](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- **Fitur**: 14 fitur klinis untuk prediksi penyakit jantung
- **Target**: Binary classification (0 = tidak sakit, 1 = sakit)

## Preprocessing
1. Hapus duplikat
2. Imputasi missing values
3. Penanganan outlier (IQR Clipping)
4. Label Encoding
5. One-Hot Encoding
6. Standarisasi fitur numerik
7. Train-Test Split (80:20)

## Cara Menjalankan
```bash
python preprocessing/automate_Suryani.py --input heart_disease_uci.csv
```

## Output
- `heart_preprocessing_train.csv`
- `heart_preprocessing_test.csv`
