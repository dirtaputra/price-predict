const tf = require('@tensorflow/tfjs');

// Data Training
const luasTanah = [150, 200, 120, 180]; // dalam m²
const jumlahKamar = [3, 4, 2, 3];
const lokasi = [1, 2, 1, 3]; // 1=Pusat Kota, 2=Pinggiran Kota, 3=Pedesaan
const harga = [500, 800, 300, 600]; // dalam juta

// Normalisasi Data
const normalize = (data) => {
  const min = Math.min(...data);
  const max = Math.max(...data);
  return data.map((val) => (val - min) / (max - min));
};

const luasTanahNorm = normalize(luasTanah);
const jumlahKamarNorm = normalize(jumlahKamar);
const lokasiNorm = normalize(lokasi);
const hargaNorm = normalize(harga);

// Membentuk Tensors
const xs = tf.tensor2d(
  luasTanahNorm.map((val, i) => [val, jumlahKamarNorm[i], lokasiNorm[i]]),
  [4, 3]
);
const ys = tf.tensor2d(hargaNorm, [4, 1]);

// Membuat Model
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [3] })); // 3 fitur
model.compile({
  optimizer: tf.train.adam(),
  loss: 'meanSquaredError',
});

// Melatih Model
(async () => {
  await model.fit(xs, ys, { epochs: 100 });
  console.log('Model telah dilatih.');

  // Prediksi
  const input = tf.tensor2d([[0.6, 0.75, 0.5]]); // LuasTanah=180, Kamar=4, Lokasi=2
  const prediction = model.predict(input);
  prediction.print(); // Output dalam bentuk normalisasi

  // Denormalisasi hasil prediksi
  const predictedPrice = prediction.dataSync()[0] * (Math.max(...harga) - Math.min(...harga)) + Math.min(...harga);
  console.log(`Prediksi Harga: ${predictedPrice} juta`);
})();