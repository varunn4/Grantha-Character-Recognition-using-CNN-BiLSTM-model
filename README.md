# Grantha Character Recognition using CNN-BiLSTM

This project presents a deep learning-based approach for recognizing characters from **Grantha script manuscripts** using a hybrid **CNN-BiLSTM architecture**. The model combines Convolutional Neural Networks (CNN) for feature extraction with Bidirectional Long Short-Term Memory (BiLSTM) networks for sequential prediction.

Grantha is an ancient script historically used in South India to write Sanskrit. Recognizing characters from handwritten manuscripts is challenging due to writing style variations, degradation of manuscripts, and complex character shapes. This project addresses these challenges using deep learning techniques.

---

# Model Architecture

The proposed architecture consists of the following components:

## 1. Preprocessing
Input manuscript images undergo preprocessing including:

- Grayscale conversion  
- Noise removal  
- Image normalization  
- Resizing for model compatibility  

## 2. CNN Feature Extractor
A Convolutional Neural Network extracts spatial features from the manuscript images.

The CNN layers learn:

- Stroke patterns  
- Character shapes  
- Local visual features  

## 3. BiLSTM Sequence Model
The extracted features are passed into a **Bidirectional Long Short-Term Memory (BiLSTM)** network to capture contextual dependencies in character sequences.

## 4. CTC Loss
The model is trained using **Connectionist Temporal Classification (CTC) Loss**, allowing sequence prediction without explicit character segmentation.

## 5. Greedy Decoder
A greedy decoding strategy is used to obtain the most probable character sequence from the model output.

---

# Dataset

The dataset used in this project consists of Grantha manuscript characters.

## Benchmark Dataset
Includes:

- Grantha consonants  
- Grantha numerals  

## Customized Dataset
A custom dataset was created containing:

- Vowels  
- Joint characters  
- Multiple writing styles  
- Variations in manuscript quality  

Dataset statistics:

- **15,000+ images**
- **140 Grantha character classes**

---

# Training and Evaluation

Three sequence models were evaluated:

- Simple RNN  
- LSTM  
- **BiLSTM**

The **BiLSTM model achieved the best performance**, showing higher accuracy and lower loss.

Training techniques used:

- Early stopping  
- Batch training  
- Model checkpointing  

---

# Repository Structure


Grantha-Character-Recognition-using-CNN-BiLSTM-model
│
├── model
│ └── crnn_model.py
│
├── training
│ └── train.py
│
├── inference
│ └── predict.py
│
├── sample_images
│
├── grantha_cnn_2.0.ipynb
├── requirements.txt
└── README.md


---

# Pretrained Models

Trained models are saved as `.pth` files.

Available architectures:

- CNN + BiLSTM  
- CNN + LSTM  
- CNN + Simple RNN  

Due to GitHub file size limits, pretrained weights are not included in this repository.

---

# Usage

## 1. Clone the repository


git clone https://github.com/varunn4/Grantha-Character-Recognition-using-CNN-BiLSTM-model.git


## 2. Navigate to the project directory


cd Grantha-Character-Recognition-using-CNN-BiLSTM-model


## 3. Install dependencies


pip install -r requirements.txt


## 4. Run prediction

Single character prediction:


python inference/predict.py


Sequence prediction:


python training/train.py


---

# Applications

- Digitization of ancient manuscripts  
- Preservation of historical documents  
- Automated transcription of Grantha texts  
- Digital humanities research  

---

# Technologies Used

- Python  
- PyTorch  
- OpenCV  
- NumPy  
- Deep Learning (CNN + BiLSTM)  
- Sequence Modeling  

---

# Future Improvements

- Transformer-based OCR models  
- Larger manuscript datasets  
- Full manuscript line recognition  
- Deployment as a web-based OCR tool  

---

# Author

**Varunn M**
**Hari Prasath J**  
Computer Science and Engineering  