## **Studi Kasus: Prediksi Harga Rumah**

### **Deskripsi**
Proyek ini bertujuan untuk memprediksi harga rumah berdasarkan tiga fitur utama:
1. **Luas tanah** (dalam m²).
2. **Jumlah kamar**.
3. **Lokasi** (kategori: pusat kota, pinggiran kota, pedesaan).

Model dibuat menggunakan **TensorFlow.js** dengan pendekatan **regresi linier multivariat**. Model dilatih menggunakan dataset kecil, dan dapat memprediksi harga rumah baru berdasarkan input data mentah.

---

## **Fitur Utama**
- **Input mentah langsung**: Sistem menerima input dalam bentuk nilai asli (tanpa normalisasi).
- **Normalisasi otomatis**: Data input baru dinormalisasi sesuai skala data training.
- **Prediksi harga**: Harga rumah diprediksi dalam juta rupiah.

---

## **Persyaratan**
- Node.js versi 14 atau lebih tinggi.
- npm (Node Package Manager) untuk mengelola dependensi.

---

## **Cara Instalasi**
Ikuti langkah-langkah berikut untuk menjalankan proyek:

1. **Clone Repository**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**
   Instal TensorFlow.js dengan npm:
   ```bash
   npm install @tensorflow/tfjs
   ```

3. **Jalankan Script**
   Untuk menjalankan script:
   ```bash
   node index.js
   ```

---

## **Cara Kerja dan Penjelasan**

### **1. Import TensorFlow.js**
```javascript
const tf = require('@tensorflow/tfjs');
```
- **TensorFlow.js** digunakan untuk membuat dan melatih model machine learning.

---

### **2. Data Training**
```javascript
const luasTanah = [150, 200, 120, 180]; // dalam m²
const jumlahKamar = [3, 4, 2, 3];
const lokasi = [1, 2, 1, 3]; // 1=Pusat Kota, 2=Pinggiran Kota, 3=Pedesaan
const harga = [500, 800, 300, 600]; // dalam juta
```
- Dataset berisi:
  - **Fitur**: `luasTanah`, `jumlahKamar`, `lokasi`.
  - **Label**: `harga`.

---

### **3. Fungsi Normalisasi**
```javascript
const normalize = (data) => {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const normalizedData = data.map((val) => (val - min) / (max - min));
  return { normalizedData, min, max };
};
```
- Normalisasi digunakan untuk mengubah data ke skala [0, 1].
- Fungsi ini mengembalikan data yang sudah dinormalisasi serta nilai **min** dan **max**.

---

### **4. Normalisasi Dataset Training**
```javascript
const luasTanahNormData = normalize(luasTanah);
const jumlahKamarNormData = normalize(jumlahKamar);
const lokasiNormData = normalize(lokasi);
const hargaNormData = normalize(harga);
```
- Normalisasi diterapkan pada seluruh dataset training.

```javascript
const luasTanahNorm = luasTanahNormData.normalizedData;
const jumlahKamarNorm = jumlahKamarNormData.normalizedData;
const lokasiNorm = lokasiNormData.normalizedData;
const hargaNorm = hargaNormData.normalizedData;

// Simpan min dan max
const luasTanahMin = luasTanahNormData.min;
const luasTanahMax = luasTanahNormData.max;
const jumlahKamarMin = jumlahKamarNormData.min;
const jumlahKamarMax = jumlahKamarNormData.max;
const lokasiMin = lokasiNormData.min;
const lokasiMax = lokasiNormData.max;
const hargaMin = hargaNormData.min;
const hargaMax = hargaNormData.max;
```
- Nilai **min** dan **max** disimpan untuk normalisasi data baru.

---

### **5. Membentuk Tensor**
```javascript
const xs = tf.tensor2d(
  luasTanahNorm.map((val, i) => [val, jumlahKamarNorm[i], lokasiNorm[i]]),
  [luasTanahNorm.length, 3]
);
const ys = tf.tensor2d(hargaNorm, [hargaNorm.length, 1]);
```
- Tensor `xs` dibentuk dari fitur (input).
- Tensor `ys` adalah label (harga).

---

### **6. Membuat Model**
```javascript
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [3] })); // 3 fitur
model.compile({
  optimizer: tf.train.adam(),
  loss: 'meanSquaredError',
});
```
- **Model**: Menggunakan **Dense Layer** untuk mempelajari hubungan antara fitur dan output.
- **Optimizer**: Adam digunakan untuk menyesuaikan bobot.
- **Loss Function**: Mean Squared Error (MSE) untuk menghitung error prediksi.

---

### **7. Melatih Model**
```javascript
(async () => {
  await model.fit(xs, ys, { epochs: 100 });
  console.log('Model telah dilatih.');
```
- **Pelatihan**: Model dilatih menggunakan dataset training selama 100 epoch.

---

### **8. Fungsi untuk Normalisasi Data Baru**
```javascript
const normalizeInput = (value, min, max) => {
  return (value - min) / (max - min);
};
```
- Fungsi ini digunakan untuk menormalisasi input baru berdasarkan nilai min dan max data training.

---

### **9. Prediksi**
```javascript
const inputBaru = {
  luasTanah: 180,
  jumlahKamar: 4,
  lokasi: 2,
};

const luasTanahNormInput = normalizeInput(inputBaru.luasTanah, luasTanahMin, luasTanahMax);
const jumlahKamarNormInput = normalizeInput(inputBaru.jumlahKamar, jumlahKamarMin, jumlahKamarMax);
const lokasiNormInput = normalizeInput(inputBaru.lokasi, lokasiMin, lokasiMax);

const inputTensor = tf.tensor2d([[luasTanahNormInput, jumlahKamarNormInput, lokasiNormInput]]);
const prediction = model.predict(inputTensor);
prediction.print();
```
- Data input mentah dinormalisasi menggunakan `normalizeInput`.
- Tensor dibuat untuk prediksi.

---

### **10. Denormalisasi Hasil Prediksi**
```javascript
const predictedPriceNorm = prediction.dataSync()[0];
const predictedPrice = predictedPriceNorm * (hargaMax - hargaMin) + hargaMin;
console.log(`Prediksi Harga: ${predictedPrice} juta`);
```
- Hasil prediksi dinormalisasi dikembalikan ke skala asli menggunakan:
  \[
  \text{harga asli} = \text{harga normalisasi} \times (\text{max} - \text{min}) + \text{min}
  \]

---

## **Output Contoh**
Setelah menjalankan script, outputnya adalah:
```
Model telah dilatih.
Tensor
    [[0.7]]  // Output dalam bentuk normalisasi
Prediksi Harga: 650 juta
```

---

## **Penjelasan Setiap Baris**

| Baris Kode                                                                                 | Penjelasan                                                                                         |
|-------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `const tf = require('@tensorflow/tfjs');`                                                 | Mengimpor TensorFlow.js untuk membangun model machine learning.                                   |
| `const luasTanah = [150, 200, 120, 180];`                                                 | Mendefinisikan data luas tanah (dalam m²).                                                       |
| `const jumlahKamar = [3, 4, 2, 3];`                                                       | Mendefinisikan data jumlah kamar.                                                                |
| `const lokasi = [1, 2, 1, 3];`                                                            | Mendefinisikan data lokasi (kategori).                                                           |
| `const harga = [500, 800, 300, 600];`                                                     | Mendefinisikan data harga rumah (dalam juta rupiah).                                             |
| `const normalize = ...`                                                                   | Fungsi untuk menormalisasi data ke skala [0, 1].                                                 |
| `const xs = tf.tensor2d(...);`                                                            | Membuat tensor dari fitur (luasTanah, jumlahKamar, lokasi).                                       |
| `const ys = tf.tensor2d(...);`                                                            | Membuat tensor dari label (harga rumah).                                                         |
| `const model = tf.sequential();`                                                          | Membuat model sequential dengan 1 layer dense.                                                   |
| `await model.fit(xs, ys, { epochs: 100 });`                                               | Melatih model menggunakan dataset training selama 100 epoch.                                     |
| `const normalizeInput = ...`                                                              | Fungsi untuk menormalisasi input baru berdasarkan data training.                                 |
| `const prediction = model.predict(inputTensor);`                                          | Membuat prediksi berdasarkan input tensor.                                                       |
| `const predictedPrice = ...`                                                              | Mengubah hasil prediksi (normalisasi) ke nilai asli menggunakan denormalisasi.                   |

---