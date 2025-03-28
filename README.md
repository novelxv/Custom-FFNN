# Custom-FFNN: Feedforward Neural Network from Scratch

---

## Deskripsi Singkat

Repositori ini berisi implementasi **Feedforward Neural Network (FFNN) dari nol (from scratch)** tanpa menggunakan library deep learning seperti TensorFlow atau PyTorch.  

Model dibangun menggunakan NumPy dan mencakup fitur seperti:
- Inisialisasi bobot (zero, uniform, normal, xavier, he)
- Fungsi aktivasi (ReLU, Sigmoid, Tanh, Softmax, GELU, Swish)
- Loss function (MSE, Binary/Categorical Cross-Entropy)
- Forward & backward propagation (batch)
- Regularisasi (L1, L2)
- Normalisasi RMSNorm
- Visualisasi distribusi bobot & gradien
- Visualisasi struktur jaringan dalam bentuk graf

---

## Struktur Direktori
```
Custom-FFNN/
├── src/
│   ├── model/                
│   │   └── ffnn.py
│   ├── layers/                
│   │   └── layer.py
│   ├── activation/            
│   ├── bonus/
│   │   ├── activation/        
│   │   ├── normalization.py  
│   │   └── init/              
│   ├── losses/              
│   ├── train/
│   │   └── train_loop.py     
│   └── utils/
│       ├── initializer.py   
│       └── visualizer.py  
│
├── notebooks/
│   └── evaluation.ipynb     
│
├── doc/
│   └── laporan.pdf            
│
├── training_history.json     
├── model_trained.pkl        
└── README.md                 
```

---

## Cara Setup & Menjalankan Program

### 1. Clone Repository

```bash
git clone https://github.com/novelxv/Custom-FFNN.git
cd Custom-FFNN
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan Notebook

```bash
cd notebooks
jupyter notebook evaluation.ipynb
```

Notebook ini mencakup:
- Training model FFNN
- Evaluasi terhadap dataset MNIST
- Eksperimen depth, width, learning rate, fungsi aktivasi, dsb
- Visualisasi bobot, gradien, dan struktur jaringan
- Perbandingan dengan `MLPClassifier` dari `sklearn`

---

## Pembagian Tugas Kelompok

**Kelompok 6:**

| Name              | NIM          | Task Description                                                                                                                                             |
|-------------------|--------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Novelya Putri Ramadhani**    | 13522096      | model/ffnn.py, layers/layer.py, losses/, utils/, activation/, notebooks/evaluation.ipynb
| **Hayya Zuhailii Kinasih**    | 13522102      | bonus/activation/, bonus/init/, bonus/normalization.py
| **Diana Tri Handayani**    | 13522104      | train/train_loop.py, utils/visualizer.py, notebooks/evaluation.ipynb

---