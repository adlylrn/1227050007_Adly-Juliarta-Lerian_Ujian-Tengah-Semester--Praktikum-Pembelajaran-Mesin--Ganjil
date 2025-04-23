# README

## Deskripsi
Repository ini berisi skrip Python untuk membangun model klasifikasi Decision Tree yang dapat membedakan buah jeruk (**orange**) dengan buah anggur (**grapefruit**) berdasarkan fitur diameter, berat, dan komponen warna (red, green, blue).

## Prasyarat
- Python 3.7+
- Library:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - joblib

Instalasi:
```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

## Tahapan Pembuatan Model
1. **Load Data**
   - Baca file `citrus.csv` ke objek `DataFrame` (`pd.read_csv('citrus.csv')`).
2. **Eksplorasi Data (EDA)**
   - Tampilkan `head()`, `info()`, `describe()`, dan checking missing values.
   - Visualisasikan histogram untuk setiap fitur.
   - Buat scatter plot `diameter` vs `weight` dengan warna sesuai label.
3. **Preprocessing**
   - Lakukan encoding label string (`orange`/`grapefruit`) menjadi numerik (0/1) menggunakan `LabelEncoder`.
   - Susun `X` (fitur: diameter, weight, red, green, blue) dan `y` (label).
4. **Split Data**
   - Bagi data menjadi training dan testing (80:20) dengan stratifikasi label.
   - Contoh: `train_test_split(..., stratify=y, random_state=42)`.
5. **Inisialisasi & Hyperparameter Tuning**
   - Definisikan `DecisionTreeClassifier(random_state=42)`.
   - Siapkan `param_grid` untuk `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
   - Jalankan `GridSearchCV` dengan 5-fold CV.
6. **Training Model**
   - Latih model terbaik (`best_estimator_`) dari Grid Search pada data training.
7. **Evaluasi Model**
   - Prediksi label dan probabilitas pada data testing.
   - Hitung dan tampilkan *classification report* (accuracy, precision, recall, F1-score).
   - Buat *confusion matrix*.
   - Hitung ROC AUC dan plot ROC Curve.
8. **Interpretasi & Visualisasi**
   - Tampilkan *feature importances*.
   - Visualisasikan pohon keputusan dengan `plot_tree`.
9. **Simpan Model**
   - Simpan model dengan `joblib.dump(best, 'decision_tree_citrus_model.pkl')` untuk deployment selanjutnya.

## Cara Menjalankan
1. Pastikan file `citrus.csv` berada di folder yang sama dengan skrip.
2. Jalankan skrip utama:
   ```bash
   python decision_tree_citrus.py
   ```
3. Hasil model yang sudah dilatih akan tersimpan di `decision_tree_citrus_model.pkl`.



