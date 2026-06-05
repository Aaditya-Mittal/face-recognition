# Custom Face Recognition System 

A complete, end-to-end face recognition pipeline built from scratch using PyTorch and Metric Learning. This project implements a custom Convolutional Neural Network (CNN) to generate 128-dimensional face embeddings, trained using Triplet Margin Loss on the Labeled Faces in the Wild (LFW) dataset.

## Features

- **Custom CNN Architecture (`FaceEmbeddingNet`)**: A lightweight, 3-layer convolutional network designed to extract spatial facial features and project them onto a 128-dimensional unit hypersphere using L2 normalization.
- **Metric Learning via Triplet Loss**: Implements dynamic triplet sampling (Anchor, Positive, Negative) to actively pull images of the same person closer together while pushing different identities apart.
- **Train/Test Pipeline**: Robust data loading pipeline with an 80/20 stratified split to ensure generalizability on unseen data.
- **SQLite Database Integration**: A built-in, lightweight database system to persistently store and retrieve the 128-dimensional embeddings of registered users.
- **Dynamic Registration & Verification**: Functions to register new users (by averaging multiple embeddings for robustness) and verify unknown faces using Euclidean distance thresholds.

## Technical Stack

- **Deep Learning Framework**: PyTorch, Torchvision
- **Data Processing**: NumPy, Scikit-learn (LFW dataset), PIL
- **Database**: SQLite, Python `pickle`

## Model Performance

The custom CNN was trained from scratch without the use of pre-trained weights (like ResNet or VGG). 
- **Training**: 30 epochs on ~4,700 cropped faces.
- **Test Set Accuracy**: **86.51%** (at a threshold of 0.8) on ~1,200 unseen images.

Given the shallow depth of the network (3 Convolutional layers) and the limited dataset size compared to industry models, an 86.51% verification accuracy serves as an excellent proof-of-concept for the underlying Triplet Loss mathematics and embedding space logic.

## Installation and Setup

To run this project locally, follow these steps:

1. **Clone the repository** (or download the files):
   ```bash
   git clone https://github.com/Aaditya-Mittal/face-recognition.git
   cd "Face Recognition"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebook**:
   Open `main.ipynb` in Jupyter Notebook or VSCode and run the cells sequentially to download the dataset, train the model, and test the face verification.

## Usage Pipeline

1. **Dataset Preparation**: `fetch_lfw_people` is used to load images, which are transformed into `[3, 128, 128]` tensors.
2. **Training**: `TripleFaceDataset` dynamically generates triplets. The model is optimized using `Adam` optimizer and `TripletMarginLoss(margin=1.0)`.
3. **Registration**: 
   - Pass 1 to N images of a person through the network.
   - Average the resulting embeddings and apply L2 normalization.
   - Serialize the final tensor and store it in SQLite alongside the person's name.
4. **Recognition**:
   - Pass an unknown face through the network.
   - Compute the Euclidean distance against all stored database embeddings.
   - If the lowest distance is below the `threshold` (e.g., 0.8), return a match.

## Future Improvements

To transition this educational proof-of-concept into a robust, real-world application, the following enhancements are recommended:
- **Pre-trained Backbones**: Swap the custom 3-layer CNN for a pre-trained ResNet-50 architecture to achieve 99%+ accuracy on real-world, unconstrained photos.
- **Scalable Database**: Migrate from SQLite to a vector database (like FAISS or PostgreSQL with pgvector) for rapid search across thousands of identities.
