# 🤟 Hand Sign Detection with CNN

A deep learning project that classifies hand signs (A-Y, excluding J and Z) from grayscale images using a Convolutional Neural Network (CNN). Built and trained using TensorFlow/Keras, this model achieves high accuracy and can be used for real-time hand sign prediction.

---

## 📁 Dataset

- Dataset Source: Kaggle : https://www.kaggle.com/datasets/ash2703/handsignimages
- Folder Structure:
  ```
  Hand_Sign/
  ├── Train/
  │   ├── a/
  │   ├── b/
  │   └── ... (until y, without j and z)
  ├── Test/
  │   ├── a/
  │   ├── b/
  │   └── ... (same as above)
  ```
- All images are grayscale hand signs.

---

## 🧠 Model Architecture

The model is a CNN (Convolutional Neural Network) built using TensorFlow/Keras. Key features:

- Input: 64x64 grayscale images
- 2 Convolutional layers + MaxPooling
- Flatten + Dense layers
- Output: Softmax activation with 24 classes (A-Y, without J and Z)

---

## 📈 Training & Evaluation

- Accuracy: ~84% on the test set
- Metrics: Precision, Recall, F1-Score per class

### Sample Confusion Matrix Report:
```
              precision    recall  f1-score   support

           A       0.93      1.00      0.96       331
           B       1.00      0.93      0.96       432
           ...
           X       0.71      0.93      0.80       267
           Y       0.80      0.71      0.75       332

    accuracy                           0.84      7172
   macro avg       0.83      0.83      0.82      7172
weighted avg       0.84      0.84      0.84      7172
```

---

## 🧪 How to Use

### 1. Install Dependencies
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn
```

### 2. Train the Model
```python
# See model_train.ipynb
model.fit(train_generator, validation_data=val_generator, epochs=20)
```

### 3. Evaluate the Model
```python
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.2%}")
```

### 4. Predict a Single Image
```python
img = preprocess_image("path/to/hand_sign.jpg")
pred = model.predict(img)
print("Predicted Label:", class_names[np.argmax(pred)])
```

### 5. Run Random Sample Prediction with Visualization
```python
# See predict_random_samples.py
```

---

## 🔍 Real-Time Webcam Prediction (Coming Soon)

A real-time webcam demo is under development using OpenCV and TensorFlow.

---

## 💾 Saving & Loading Model
```python
model.save("hand_sign_model.h5")
model = load_model("hand_sign_model.h5")
```

---

## 🛠️ Tools & Libraries

- TensorFlow / Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-learn

---


