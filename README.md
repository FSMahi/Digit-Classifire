## MNIST Digit Classifier with ANN & CNN

````markdown
# 🧠 Handwritten Digit Classifier — ANN vs CNN (MNIST Dataset)

This project implements and compares two deep learning models — a **Dense Neural Network (ANN)** and a **Convolutional Neural Network (CNN)** — to classify handwritten digits (0–9) using the MNIST dataset. Regularization methods such as Dropout, Batch Normalization, and L2 weight decay are applied to evaluate model generalization.

---

## 📌 Project Objectives

- Load and preprocess the MNIST dataset.
- Build and train two models:
  - **Artificial Neural Network (ANN)**
  - **Convolutional Neural Network (CNN)**
- Apply regularization:
  - Dropout
  - Batch Normalization
  - L2 Regularization (Weight Decay)
- Compare performance and visualize metrics.
- Evaluate final accuracy on test data.

---

## 📂 Dataset: MNIST

- 60,000 training images
- 10,000 test images
- 28×28 grayscale digit images (0–9)

Loaded via:
```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
````

---

## 🧠 Model 1: Artificial Neural Network (ANN)

### 🔸 Architecture:

* Flatten(28×28 → 784)
* Dense(256, ReLU) → BatchNorm → Dropout(0.3)
* Dense(128, ReLU) → BatchNorm → Dropout(0.3)
* Dense(64, ReLU) → BatchNorm
* Output: Dense(10, Softmax)

### 🔸 Regularized Version:

Same architecture but with **L2 kernel regularization** applied to each Dense layer.

### 🔹 Training:

* Optimizer: `Adam`
* Loss: `categorical_crossentropy`
* Epochs: 10–20
* Validation Split: 0.1

---

## 🧠 Model 2: Convolutional Neural Network (CNN)

### 🔸 Architecture:

* Input: Reshaped to (28, 28, 1)
* Conv2D(32) → BatchNorm → MaxPooling → Dropout(0.25)
* Conv2D(64) → BatchNorm → MaxPooling → Dropout(0.25)
* Flatten
* Dense(128, ReLU) → BatchNorm → Dropout(0.5)
* Output: Dense(10, Softmax)

### 🔸 Regularized Version:

Conv2D and Dense layers use **L2 kernel regularization**.

---

## 🔁 Comparison: ANN vs CNN

| Metric               | ANN (Basic) | ANN + L2 | CNN (Basic) | CNN + L2 |
| -------------------- | ----------- | -------- | ----------- | -------- |
| Final Train Accuracy | \~98%       | \~97.5%  | \~99%       | \~98.8%  |
| Final Val Accuracy   | \~97–98%    | \~97.5%  | \~98.5–99%  | \~98.6%  |
| Test Accuracy        | \~97.8%     | \~97.5%  | \~98.8%     | \~98.6%  |
| Params               | \~300k+     | Same     | \~1.2M+     | Same     |

✅ **CNN consistently outperforms ANN**, especially on unseen data.

---

## 📈 Visualizations

* **Training vs Validation Accuracy** (per model)
* **Training vs Validation Loss**
* **Class Distribution** using Seaborn
* Optional:

  * Confusion Matrix
  * Sample Misclassified Digits

---

## 🧪 Evaluation

Use `model.evaluate()` to check test performance:

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", test_acc)
print("Test Loss:", test_loss)
```

---

## 📊 Regularization Impact

* **Dropout** reduces overfitting by randomly disabling neurons during training.
* **Batch Normalization** stabilizes and speeds up training.
* **L2 Regularization** penalizes large weights, improving generalization.

> Using all three techniques together provides the best balance of performance and generalization.

---

## 🚀 How to Run

Open the Colab notebook:

```bash
https://colab.research.google.com/drive/YOUR-NOTEBOOK-LINK
```

### Requirements (in Colab by default):

* TensorFlow 2.x
* Matplotlib, Seaborn
* Pandas, NumPy

---

## ✍️ Author

**Your Name**
Dept. of ICT, Jahangirnagar University
🔗 GitHub: \[your-github-link]
📧 Email: \[your-email]

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

* [TensorFlow Datasets](https://www.tensorflow.org/datasets)
* [Keras Documentation](https://keras.io/)
* Google Colab for cloud-based GPU training

```

---

```
