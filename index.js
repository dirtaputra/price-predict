const tf = require('@tensorflow/tfjs');

// Data Training
const luasTanah = [150, 200, 120, 180]; // dalam m²
const jumlahKamar = [3, 4, 2, 3];
const lokasi = [1, 2, 1, 3]; // 1=Pusat Kota, 2=Pinggiran Kota, 3=Pedesaan
const harga = [500, 800, 300, 600]; // dalam juta

// Normalisasi Data dengan mengembalikan min dan max
const normalize = (data) => {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const normalizedData = data.map((val) => (val - min) / (max - min));
  return { normalizedData, min, max };
};

// Normalisasi Data dan simpan min dan max
const luasTanahNormData = normalize(luasTanah);
const jumlahKamarNormData = normalize(jumlahKamar);
const lokasiNormData = normalize(lokasi);
const hargaNormData = normalize(harga);

// Data yang telah dinormalisasi
const luasTanahNorm = luasTanahNormData.normalizedData;
const jumlahKamarNorm = jumlahKamarNormData.normalizedData;
const lokasiNorm = lokasiNormData.normalizedData;
const hargaNorm = hargaNormData.normalizedData;

// Simpan min dan max untuk setiap fitur
const luasTanahMin = luasTanahNormData.min;
const luasTanahMax = luasTanahNormData.max;

const jumlahKamarMin = jumlahKamarNormData.min;
const jumlahKamarMax = jumlahKamarNormData.max;

const lokasiMin = lokasiNormData.min;
const lokasiMax = lokasiNormData.max;

const hargaMin = hargaNormData.min;
const hargaMax = hargaNormData.max;

// Membentuk Tensors
const xs = tf.tensor2d(
  luasTanahNorm.map((val, i) => [val, jumlahKamarNorm[i], lokasiNorm[i]]),
  [luasTanahNorm.length, 3]
);
const ys = tf.tensor2d(hargaNorm, [hargaNorm.length, 1]);

// Membuat Model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [3] })); // 3 fitur
model.compile({
  optimizer: tf.train.adam(),
  loss: 'meanSquaredError',
});

// Fungsi untuk menormalisasi input baru
const normalizeInput = (value, min, max) => {
  return (value - min) / (max - min);
};

// Melatih Model dan Melakukan Prediksi
(async () => {
  await model.fit(xs, ys, { epochs: 100 });
  console.log('Model telah dilatih.');

  // Data Input Mentah
  const inputBaru = {
    luasTanah: 180,
    jumlahKamar: 4,
    lokasi: 2, // 1=Pusat Kota, 2=Pinggiran Kota, 3=Pedesaan
  };

  // Normalisasi Data Input Baru
  const luasTanahNormInput = normalizeInput(inputBaru.luasTanah, luasTanahMin, luasTanahMax);
  const jumlahKamarNormInput = normalizeInput(inputBaru.jumlahKamar, jumlahKamarMin, jumlahKamarMax);
  const lokasiNormInput = normalizeInput(inputBaru.lokasi, lokasiMin, lokasiMax);

  // Membuat Tensor Input
  const inputTensor = tf.tensor2d([[luasTanahNormInput, jumlahKamarNormInput, lokasiNormInput]]);
  
  // Prediksi
  const prediction = model.predict(inputTensor);
  prediction.print(); // Output dalam bentuk normalisasi

  // Denormalisasi hasil prediksi
  const predictedPriceNorm = prediction.dataSync()[0];
  const predictedPrice = predictedPriceNorm * (hargaMax - hargaMin) + hargaMin;
  console.log(`Prediksi Harga: ${predictedPrice} juta`);
})();
