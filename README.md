# "Iris Dataset" Analysis using Machine Learning Techniques (XGBoost/Gradient Boosted Decision Trees)

## **Introduction**
In this project, I embarked on a journey into the Iris dataset, a classic in the field of machine learning. The dataset comprises 150 samples of iris flowers, each characterized by its own size of features like sepal and petal. The goal of this project is to build and optimize a machine learning model for accurate classification of iris species. I've explicitly implemented **two versions** of the model using:
1. **Gradient Boosted Decision Trees (GBDT)**: A step-by-step manual implementation to demonstrate the core concepts of boosting.
2. **XGBoost**: A library implementation that offers state-of-the-art performance for gradient boosting.

## **Project Workflow**

### **1. Dataset Overview**
- The Iris dataset is publicly available in the `scikit-learn` library.
- It contains:
  - **150 samples** divided equally among three classes (Setosa, Versicolor, Virginica).
  - **4 features** per sample (sepal length, sepal width, petal length, petal width).

### **2. Data Preprocessing**
- **One-Hot Encoding**:
  - Multi-class labels (`0`, `1`, `2`) are converted into one-hot encoded vectors for GBDT implementation.
  - For example, class `0` becomes `[1, 0, 0]`, class `1` becomes `[0, 1, 0]`, and class `2` becomes `[0, 0, 1]`.
- **Train-Test Split**:
  - Data is split into training (80%) and testing (20%) sets using `train_test_split` from `scikit-learn`.

### **3. Gradient Boosted Decision Trees (GBDT)**

#### **Core Concepts**:
- **Gradient Boosting**:
  - Residuals (errors) are iteratively computed to improve predictions.
  - Weak learners (decision trees) are sequentially trained on these residuals.
- **Decision Trees**:
  - Each tree learns to predict the residuals for a specific class.
  - Trees are constrained to a small depth (e.g., `max_depth=3`) to ensure they remain weak learners.
- **Multi-Class Handling**:
  - Separate decision trees are trained for each class.
  - Predictions are combined, and the class with the highest probability is selected.

#### **Implementation Details**:
1. **Residual Calculation**:
   - Residuals are computed as:
     ```
     residuals = y_true - y_pred
     ```
2. **Tree Training**:
   - Decision trees are trained on the residuals for each class.
3. **Prediction Update**:
   - Predictions are updated iteratively using:
     ```
     y_pred += learning_rate * tree.predict(X)
     ```
4. **Final Prediction**:
   - The final class is determined using `argmax` over the combined predictions:
     ```
     final_predictions = np.argmax(y_pred, axis=1)
     ```

---

### **4. XGBoost**

#### **Core Features**:
- **Optimized Gradient Boosting**:
  - XGBoost automates and optimizes the gradient boosting process.
- **Regularization**:
  - Adds L1/L2 penalties to prevent overfitting.
- **Advanced Split Techniques**:
  - Efficiently chooses splits using techniques like Weighted Quantile Sketch.
- **Built-in Multi-Class Support**:
  - Handles multi-class problems without manual one-hot encoding.

#### **Implementation Steps**:
1. Initialize the XGBoost DMatrix from its library:
   ```python
   train = xgb.DMatrix(X_train, label=y_train)
   test = xgb.DMatrix(X_test, label=y_test)
   ```
2. Train the model on the training data:
   ```python
   model = xgb.train(param, train, epochs)
   ```
3. Evaluate on test data:
   ```python
   predictions = model.predict(test)
   ```
4. Calculate accuracy:
   ```python
   accuracy = accuracy_score(y_test, y_pred)
   ```

---

## **Key Concepts Learned**
1. **Gradient Boosting**:
   - Sequentially improve model predictions by focusing on residuals (errors).
2. **Decision Trees**:
   - Used as weak learners in gradient boosting.
3. **Ensemble Learning**:
   - Combines multiple weak learners to create a strong predictive model.
4. **Learning Rate**:
   - Controls the step size of updates to avoid overfitting.
5. **Multi-Class Classification**:
   - Handled using one-hot encoding for GBDT and softmax for XGBoost.
6. **Regularization** (via XGBoost):
   - Prevents overfitting through L1/L2 penalties.

---

## **Results**
- **Gradient Boosted Decision Trees (GBDT)**:
  - Accuracy on test data: ~85%
  - Key takeaway: Provides insights into the inner workings of gradient boosting.
- **XGBoost**:
  - Accuracy on test data: ~95%
  - Key takeaway: Optimized implementation significantly outperforms manual GBDT.

---

## **How to Run the Code**

### **Dependencies**:
Install the required libraries:
```bash
pip install numpy scikit-learn xgboost
```

### **Steps**:
1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/iris-analysis.git
   cd iris-analysis
   ```
2. Run the notebook:
   ```bash
   jupyter notebook Iris_Analysis.ipynb
   ```
3. Explore the results:
   - GBDT implementation is explained step-by-step.
   - XGBoost implementation demonstrates the power of the library.

---

## **Conclusion**
This project demonstrates both a **manual implementation of Gradient Boosting** and the use of **XGBoost** to classify the Iris dataset effectively. It highlights:
- The fundamental concepts behind boosting.
- The power of XGBoost for real-world machine learning tasks.

Feel free to explore, experiment, and adapt the code for your own projects!

