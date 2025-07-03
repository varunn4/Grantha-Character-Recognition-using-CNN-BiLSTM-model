#CHARACTER RECOGNITION IN GRANTHA SCRIPT MANUSCRIPTS USING A HYBRID CNN-BILSTM MODEL


This project presents a deep learning-based approach for recognizing characters in Grantha script manuscripts. The proposed model leverages a hybrid CNN-BiLSTM architecture, combining the strengths of convolutional neural networks (CNNs) for feature extraction and bidirectional long short-term memory (BiLSTM) networks for sequential prediction.
Model Architecture
- Preprocessing: Input images undergo preprocessing using standard image processing techniques.
- CNN: A CNN is employed for character recognition, extracting features from the input images.
- BiLSTM: A BiLSTM network is used for sequential prediction, capturing contextual dependencies in the character sequences.
- CTC Loss: The Connectionist Temporal Classification (CTC) loss function is utilized for sequential prediction.
- Greedy Decoder: A greedy decoder is used to obtain the most likely output.

Dataset
- Benchmark Dataset: The model is initially trained on a benchmark dataset consisting of consonants and numerals of the Grantha script.
- Customized Dataset: A comprehensive customized dataset is created for vowels and joint letters, featuring various styles and variations.

Training and Evaluation
- Model Comparison: The performance of BiLSTM, LSTM, and Simple RNN models is compared, with BiLSTM achieving superior accuracy and lower loss.
- Early Stopping: Early stopping criteria are employed to prevent overfitting.



Usage
- Pre-trained Models: Pre-trained models are saved as .pth files for each architecture (BiLSTM, LSTM, Simple RNN).
- Testing: Users can test the models using single_pred.py for single character recognition and seq.py for sequential character recognition.
- Label Dictionary: A label dictionary is used for predicting the output.

Getting Started
1. Clone the repository and navigate to the project directory.
2. Install the required dependencies.
3. Use single_pred.py or seq.py to test the pre-trained models.
