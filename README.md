# Grantha Character Recognition using CNN-BiLSTM

This project presents a deep learning-based approach for recognizing characters from **Grantha script manuscripts** using a hybrid **CNN-BiLSTM architecture**. The model combines the strengths of **Convolutional Neural Networks (CNN)** for feature extraction and **Bidirectional Long Short-Term Memory (BiLSTM)** networks for sequence prediction.

Grantha is an ancient script historically used in South India to write Sanskrit. Recognizing characters from handwritten manuscripts is challenging due to variations in writing styles, degradation of manuscripts, and complex character structures. This project aims to address these challenges using deep learning techniques.

---

# Model Architecture

The proposed architecture consists of the following components:

## 1. Preprocessing
Input manuscript images undergo preprocessing including:

- Grayscale conversion
- Noise removal
- Image normalization
- Image resizing for model compatibility

## 2. CNN Feature Extraction
A **Convolutional Neural Network (CNN)** extracts spatial features from manuscript images. The CNN learns:

- Stroke patterns
- Character shapes
- Local structural features

## 3. BiLSTM Sequence Modeling
The extracted features are passed to a **Bidirectional Long Short-Term Memory (BiLSTM)** network that captures contextual dependencies in character sequences.

## 4. CTC Loss
The model is trained using **Connectionist Temporal Classification (CTC) Loss**, allowing sequence prediction without explicit character segmentation.

## 5. Greedy Decoder
A greedy decoding strategy is used to obtain the most probable output sequence from the network.

---

# Dataset

The dataset used for this project contains Grantha manuscript characters.

## Benchmark Dataset
Includes:

- Grantha consonants
- Grantha numerals

## Customized Dataset
A custom dataset was created to include:

- Vowels
- Joint characters
- Multiple handwriting styles
- Variations in manuscript quality

Dataset statistics:

- **15,000+ character images**
- **140 Grantha character classes**

---

# Training and Evaluation

Different sequence models were evaluated:

- Simple RNN
- LSTM
- **BiLSTM**

Among these, the **BiLSTM model achieved the best performance**, providing improved accuracy and lower loss.

Training techniques used:

- Early stopping
- Batch training
- Model checkpointing

---

# Repository Structure


Grantha-Character-Recognition-using-CNN-BiLSTM-model
│
├── model
│ ├── seq.ipynb
│ └── single_pred.ipynb
│
├── grantha_cnn_2.0.ipynb
├── requirements.txt
├── README.md
└── .gitignore


### File Description

**grantha_cnn_2.0.ipynb**

Main notebook containing:

- Data preprocessing
- CNN-BiLSTM model architecture
- Training pipeline
- Model evaluation

**model/single_pred.ipynb**

Notebook used for **single character prediction** from Grantha manuscript images.

**model/seq.ipynb**

Notebook used for **sequential character prediction** for recognizing character sequences.

---

# Usage

## 1. Clone the repository


git clone https://github.com/varunn4/Grantha-Character-Recognition-using-CNN-BiLSTM-model.git


## 2. Navigate to the project directory


cd Grantha-Character-Recognition-using-CNN-BiLSTM-model


## 3. Install dependencies


pip install -r requirements.txt


## 4. Run the notebooks

Open the notebooks using **Jupyter Notebook or VS Code**.

### Model Training

Run:


grantha_cnn_2.0.ipynb


### Single Character Prediction

Run:


model/single_pred.ipynb


### Sequential Character Prediction

Run:


model/seq.ipynb


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

# Author

**Varunn M**  

**Hari Prasath J**
