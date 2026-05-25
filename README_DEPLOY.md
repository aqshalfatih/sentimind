# SentiMind TensorFlow.js untuk Vercel

Versi ini sudah tidak memakai Flask dan tidak memakai TensorFlow Python saat deploy.
Model LSTM dijalankan langsung di browser menggunakan TensorFlow.js.

## Struktur penting

- `index.html` = halaman utama statis
- `js/sentimind_tfjs.js` = preprocessing, tokenisasi, LSTM inference, dan render hasil
- `model/*.bin` = bobot model hasil ekstraksi dari `model_lstm_sentimen_utbk.h5`
- `model/tokenizer_config.json` = tokenizer, normalisasi kata, stopword, label, dan konfigurasi model
- `media/icon-sentimind.png` = logo

## Cara deploy ke Vercel

1. Upload isi folder ini ke GitHub.
2. Buka Vercel dan pilih **New Project**.
3. Import repository GitHub tersebut.
4. Framework preset pilih **Other**.
5. Build command kosongkan.
6. Output directory kosongkan atau isi `.`.
7. Deploy.

## Catatan penting

- Jangan upload lagi file `app.py`, `requirements.txt`, `.h5`, `.pkl`, atau folder `.git` lama ke project Vercel ini.
- File `requirements.txt` lama berisi `tensorflow`, sehingga Vercel mencoba membuat Python Function besar.
- Di versi ini, model diproses di sisi browser sehingga tidak ada serverless function Python.
- `max_len` disesuaikan menjadi `50` karena konfigurasi model `.h5` memiliki input shape `[batch, 50]`.
- Buka lewat Vercel/Live Server. Jika dibuka langsung dengan `file://`, browser biasanya memblokir `fetch()` ke file model.


## Optimasi loading model

Versi ini sudah diperbaiki agar loading terasa lebih cepat:

- bobot model dimuat paralel, bukan satu per satu;
- file tokenizer dan embedding dipreload dari awal;
- model dimuat di background setelah halaman tampil;
- jika pengguna langsung menekan tombol prediksi, sistem akan menunggu model selesai dimuat lalu lanjut memproses;
- script TensorFlow.js dipin ke versi stabil agar tidak berubah-ubah karena `@latest`.
