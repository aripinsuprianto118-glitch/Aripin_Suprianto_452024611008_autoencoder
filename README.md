# Tugas Autoencoder Fashion-MNIST

## 1. Identitas
- **Nama:** Aripin Suprianto
- **NIM:** 452024611008

## 2. Versi Python dan Library yang Digunakan
- **Python Version:** 3.10 / 3.12 (atau lebih baru)
- **Libraries:**
  - `torch` (PyTorch)
  - `torchvision`
  - `matplotlib`

## 3. Cara Menjalankan Training di Kaggle
1. Buka Kaggle dan buat Notebook baru.
2. Upload file `autoencoder-fashion-mnist.ipynb` ke dalam Notebook.
3. Pastikan memilih akselerator GPU (opsional, untuk mempercepat training) atau biarkan standar CPU.
4. Jalankan semua cell (Run All) pada notebook.
5. Setelah proses training selesai, unduh (download) file model bobot yang dihasilkan, yaitu `autoencoder_fashion_mnist.pth` dan `decoder_fashion_mnist.pth`.

## 4. Cara Menjalankan Rekonstruksi dari Terminal (Lokal)
1. Pastikan Anda telah menginstal library yang dibutuhkan dengan menjalankan perintah `pip install torch torchvision matplotlib` di terminal.
2. Pindahkan file model (`.pth`) yang telah diunduh dari Kaggle ke dalam folder yang sama dengan file skrip Python (`reconstruct.py` dan `generate_from_decoder.py`).
3. Buka Terminal (Command Prompt / PowerShell), lalu arahkan direktori ke folder tugas tersebut.
4. Jalankan perintah eksekusi sesuai dengan program yang ingin diuji.

## 5. Contoh Perintah Terminal
**Opsi A: Rekonstruksi Menggunakan Autoencoder Penuh**
```bash
python reconstruct.py --model autoencoder_fashion_mnist.pth --index 25