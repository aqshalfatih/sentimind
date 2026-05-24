let sentimindConfig = null;
let sentimindWeights = null;
let sentimindReady = false;

const statusEl = () => document.getElementById("modelStatus");
const submitBtnEl = () => document.getElementById("submitBtn");

function setModelStatus(message, isError = false) {
  const el = statusEl();
  if (!el) return;
  el.textContent = message;
  el.classList.toggle("text-red-300", isError);
  el.classList.toggle("text-gray-500", !isError);
}

async function fetchJson(path) {
  const response = await fetch(path, { cache: "force-cache" });
  if (!response.ok) throw new Error(`Gagal memuat ${path}`);
  return response.json();
}

async function loadTensor(path, shape) {
  const response = await fetch(path, { cache: "force-cache" });
  if (!response.ok) throw new Error(`Gagal memuat bobot ${path}`);
  const buffer = await response.arrayBuffer();
  const values = new Float32Array(buffer);
  return tf.tensor(values, shape, "float32");
}

async function loadSentimindModel() {
  const button = submitBtnEl();
  if (button) {
    button.disabled = true;
    button.classList.add("opacity-70", "cursor-not-allowed");
  }

  try {
    setModelStatus("Memuat TensorFlow.js dan model LSTM...");
    await tf.ready();

    sentimindConfig = await fetchJson("./model/tokenizer_config.json");
    const manifest = sentimindConfig.weights;

    sentimindWeights = {
      embedding: await loadTensor(`./model/${manifest.embedding.path}`, manifest.embedding.shape),
      lstmKernel: await loadTensor(`./model/${manifest.lstm_kernel.path}`, manifest.lstm_kernel.shape),
      lstmRecurrentKernel: await loadTensor(`./model/${manifest.lstm_recurrent_kernel.path}`, manifest.lstm_recurrent_kernel.shape),
      lstmBias: await loadTensor(`./model/${manifest.lstm_bias.path}`, manifest.lstm_bias.shape),
      denseKernel: await loadTensor(`./model/${manifest.dense_kernel.path}`, manifest.dense_kernel.shape),
      denseBias: await loadTensor(`./model/${manifest.dense_bias.path}`, manifest.dense_bias.shape),
      outputKernel: await loadTensor(`./model/${manifest.output_kernel.path}`, manifest.output_kernel.shape),
      outputBias: await loadTensor(`./model/${manifest.output_bias.path}`, manifest.output_bias.shape),
    };

    sentimindReady = true;
    if (button) {
      button.disabled = false;
      button.classList.remove("opacity-70", "cursor-not-allowed");
      button.innerHTML = '<i class="fas fa-search"></i> Prediksi Sentimen';
    }
    setModelStatus("Model siap digunakan di browser tanpa backend Flask.");
  } catch (error) {
    console.error(error);
    setModelStatus("Model gagal dimuat. Jalankan melalui Live Server/Vercel, bukan dibuka langsung dari file explorer.", true);
  }
}

function normalizeWords(text) {
  let normalized = String(text || "")
    .replace(/try out/g, "tryout")
    .replace(/try-out/g, "tryout");

  return normalized
    .split(/\s+/)
    .filter(Boolean)
    .map((word) => sentimindConfig.normalization_dict[word] || word)
    .join(" ");
}

function cleanForLabeling(text) {
  let cleaned = String(text || "")
    .toLowerCase()
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#039;/g, "'")
    .replace(/https?:\/\/\S+|www\.\S+/g, " ")
    .replace(/@\w+/g, " ")
    .replace(/#/g, " ")
    .replace(/[^a-zA-Z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

  return normalizeWords(cleaned);
}

function cleanText(text) {
  const stopwords = new Set(sentimindConfig.stopwords);
  return cleanForLabeling(text)
    .split(/\s+/)
    .filter((word) => word && !stopwords.has(word))
    .join(" ");
}

function textToSequence(cleanedText) {
  const wordIndex = sentimindConfig.word_index;
  const oovIndex = sentimindConfig.oov_index || 1;
  const numWords = sentimindConfig.num_words || 10000;

  return cleanedText
    .split(/\s+/)
    .filter(Boolean)
    .map((word) => {
      let index = wordIndex[word];
      if (index === undefined || index === null || index >= numWords) index = oovIndex;
      return index;
    });
}

function padPost(sequence) {
  const maxLen = sentimindConfig.max_len || 50;
  const padded = new Array(maxLen).fill(0);
  const limit = Math.min(sequence.length, maxLen);
  for (let i = 0; i < limit; i += 1) padded[i] = sequence[i];
  return padded;
}

function postProcessSentiment(originalText, label, confidence) {
  const text = cleanForLabeling(originalText);

  const strongNegative = [
    "tidak lolos", "belum lolos", "kurang lolos",
    "tidak lulus", "belum lulus", "kurang lulus",
    "gagal lolos", "gagal lulus",
    "ga lolos", "gak lolos", "nggak lolos", "ngga lolos",
    "ga lulus", "gak lulus", "nggak lulus", "ngga lulus",
    "tidak masuk ptn", "belum masuk ptn",
    "tidak diterima", "belum diterima", "ditolak",
    "gagal utbk", "gagal snbt", "gagal masuk", "gagal ptn",
    "tidak siap", "belum siap", "kurang siap",
    "tidak yakin", "belum yakin",
    "takut gagal", "takut tidak lolos", "takut tidak lulus",
    "takut tidak diterima", "cemas", "stres", "panik",
    "khawatir", "ragu",
    "skor rendah", "nilai rendah", "skor turun", "nilai turun",
    "jauh dari target", "masih rendah", "masih kurang",
    "tidak paham", "belum paham", "susah paham",
    "sulit paham", "soal susah", "materi susah",
    "mau menyerah", "ingin menyerah",
    "mental down", "capek banget", "pusing banget"
  ];

  const strongPositive = [
    "lolos utbk", "lolos snbt", "lolos ptn",
    "lulus utbk", "lulus snbt", "lulus ptn",
    "berhasil lolos", "berhasil lulus",
    "akhirnya lolos", "akhirnya lulus",
    "diterima ptn", "masuk ptn",
    "skor naik", "nilai naik", "skor aman", "nilai aman"
  ];

  const positiveException = [
    "jangan takut", "jangan menyerah", "tetap semangat",
    "pasti bisa", "yakin bisa", "semoga lulus", "semoga lolos"
  ];

  if (strongNegative.some((phrase) => text.includes(phrase))) {
    return { label: "negatif", confidence: Math.max(confidence, 85.0) };
  }

  if (strongPositive.some((phrase) => text.includes(phrase))) {
    return { label: "positif", confidence: Math.max(confidence, 80.0) };
  }

  if (positiveException.some((phrase) => text.includes(phrase))) {
    return { label: "positif", confidence: Math.max(confidence, 75.0) };
  }

  return { label, confidence };
}

async function predictSentiment(comment) {
  const clean = cleanText(comment);
  const sequence = padPost(textToSequence(clean));
  const labels = sentimindConfig.labels;

  const probsTensor = tf.tidy(() => {
    const input = tf.tensor1d(sequence, "int32");
    const embedded = tf.gather(sentimindWeights.embedding, input);

    const units = 64;
    let h = tf.zeros([1, units]);
    let c = tf.zeros([1, units]);

    for (let t = 0; t < sequence.length; t += 1) {
      const xt = embedded.slice([t, 0], [1, 128]);
      const z = xt
        .matMul(sentimindWeights.lstmKernel)
        .add(h.matMul(sentimindWeights.lstmRecurrentKernel))
        .add(sentimindWeights.lstmBias);

      const [zi, zf, zc, zo] = tf.split(z, 4, 1);
      const i = tf.sigmoid(zi);
      const f = tf.sigmoid(zf);
      const candidate = tf.tanh(zc);
      const o = tf.sigmoid(zo);

      c = f.mul(c).add(i.mul(candidate));
      h = o.mul(tf.tanh(c));
    }

    const dense = tf.relu(h.matMul(sentimindWeights.denseKernel).add(sentimindWeights.denseBias));
    const logits = dense.matMul(sentimindWeights.outputKernel).add(sentimindWeights.outputBias);
    return tf.softmax(logits).reshape([3]);
  });

  const probabilities = Array.from(await probsTensor.data());
  probsTensor.dispose();

  let labelIndex = 0;
  for (let i = 1; i < probabilities.length; i += 1) {
    if (probabilities[i] > probabilities[labelIndex]) labelIndex = i;
  }

  let label = labels[labelIndex];
  let confidence = probabilities[labelIndex] * 100;
  const processed = postProcessSentiment(comment, label, confidence);
  label = processed.label;
  confidence = Math.round(processed.confidence * 100) / 100;

  return { label, confidence, clean };
}

function escapeText(text) {
  const div = document.createElement("div");
  div.textContent = text || "-";
  return div.innerHTML;
}

function renderResult(comment, result) {
  const hasil = document.getElementById("hasil");
  const card = document.getElementById("sentimentCard");
  const iconBox = document.getElementById("sentimentIconBox");
  const icon = document.getElementById("sentimentIcon");
  const badge = document.getElementById("sentimentBadge");
  const label = document.getElementById("resultLabel");
  const desc = document.getElementById("resultDescription");
  const confidence = document.getElementById("resultConfidence");
  const bar = document.getElementById("confidenceBar");
  const original = document.getElementById("resultOriginal");
  const clean = document.getElementById("resultClean");

  const view = {
    positif: {
      card: "result-positive rounded-2xl p-5",
      iconBox: "w-12 h-12 rounded-xl bg-green-400/15 flex items-center justify-center mb-4",
      icon: "fa-solid fa-face-smile text-green-300 text-xl",
      badge: "text-sm text-green-200 font-bold uppercase tracking-wider",
      label: "text-4xl md:text-5xl font-black text-green-300 mt-2",
      title: "Positif",
      description: "Komentar menunjukkan motivasi, kesiapan, harapan, atau optimisme."
    },
    negatif: {
      card: "result-negative rounded-2xl p-5",
      iconBox: "w-12 h-12 rounded-xl bg-red-400/15 flex items-center justify-center mb-4",
      icon: "fa-solid fa-face-frown text-red-300 text-xl",
      badge: "text-sm text-red-200 font-bold uppercase tracking-wider",
      label: "text-4xl md:text-5xl font-black text-red-300 mt-2",
      title: "Negatif",
      description: "Komentar menunjukkan kecemasan, tekanan, ketakutan, atau ketidaksiapan."
    },
    netral: {
      card: "result-neutral rounded-2xl p-5",
      iconBox: "w-12 h-12 rounded-xl bg-yellow-400/15 flex items-center justify-center mb-4",
      icon: "fa-solid fa-circle-info text-yellow-300 text-xl",
      badge: "text-sm text-yellow-100 font-bold uppercase tracking-wider",
      label: "text-4xl md:text-5xl font-black text-yellow-300 mt-2",
      title: "Netral",
      description: "Komentar bersifat informatif atau tidak menunjukkan emosi yang kuat."
    }
  }[result.label] || view.netral;

  card.className = view.card;
  iconBox.className = view.iconBox;
  icon.className = view.icon;
  badge.className = view.badge;
  label.className = view.label;
  label.textContent = view.title;
  desc.textContent = view.description;
  confidence.textContent = `${result.confidence}%`;
  bar.style.width = `${Math.min(100, Math.max(0, result.confidence))}%`;
  original.innerHTML = escapeText(comment);
  clean.innerHTML = escapeText(result.clean || "-");

  hasil.classList.remove("hidden");
  window.location.hash = "hasil";
}

function setupPredictionForm() {
  const form = document.getElementById("predictionForm");
  const input = document.getElementById("commentInput");
  const button = submitBtnEl();

  if (!form || !input || !button) return;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const comment = input.value.trim();
    if (!comment) return;

    if (!sentimindReady) {
      setModelStatus("Model masih dimuat. Coba lagi setelah status model siap.", true);
      return;
    }

    button.disabled = true;
    button.classList.add("opacity-80", "cursor-not-allowed");
    button.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Memproses prediksi...';

    try {
      const result = await predictSentiment(comment);
      renderResult(comment, result);
      setModelStatus("Prediksi selesai. Semua proses berjalan di browser.");
    } catch (error) {
      console.error(error);
      setModelStatus("Prediksi gagal diproses. Cek console browser untuk detail error.", true);
    } finally {
      button.disabled = false;
      button.classList.remove("opacity-80", "cursor-not-allowed");
      button.innerHTML = '<i class="fas fa-search"></i> Prediksi Sentimen';
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  setupPredictionForm();
  loadSentimindModel();
});
