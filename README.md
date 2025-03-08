# CS4700-Artificial-Intelligence
# Bonus Assignment - CS4700 Artificial Intelligence

## Student Information
- **Name:** Niharika Goud Cika
- **Student ID:** 700773687
- **Course:** CS4700 - Artificial Intelligence
- **Semester:** Spring 2025
- **University:** University of Central Missouri  

## Overview
This repository contains solutions for **Bonus Assignment 2** in **CS4700 - Artificial Intelligence**. The assignment covers different AI and ML techniques including **Linear Regression, K-Means Clustering, Neural Networks, and Reinforcement Learning**.

The following exercises were implemented:
- **Exercise 1:** House Price Prediction using Linear Regression
- **Exercise 2:** K-Means Clustering
- **Exercise 3:** Simple Neural Network for Regression
- **Exercise 4:** Updating the State in the GridWorld Environment  



## **Exercise 1: House Price Prediction using Linear Regression**
### **Description**
This exercise implements **Linear Regression** using the **California Housing dataset** to predict house prices.

### **Code Explanation**
1. **Dataset Loading:**  
   - The dataset is loaded using `fetch_california_housing()`. It contains **features like median income, house age, and location-based information**, which are used to predict house prices.
  
2. **Train-Test Split (80/20):**  
   - We split the data into **80% training data** and **20% testing data** to evaluate the model.
   - We use `random_state=42` to ensure reproducibility.
  
3. **Model Training:**  
   - We create a **Linear Regression** model using `LinearRegression()`.
   - The model is trained using `model.fit(X_train, y_train)`, which finds the best fit line for the training data.

4. **Predictions & Evaluation:**
   - We use `model.predict(X_test)` to generate predicted house prices.
   - Performance metrics:
     - **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
     - **R² Score**: Measures how well the model explains the variance in house prices.

### **Why These Blanks Were Filled?**
- `test_size=0.2` → Ensures **80% training and 20% testing split**, a common practice in machine learning.
- `model.fit(X_train, y_train)` → Trains the **Linear Regression** model by finding the optimal weights.
- `model.predict(X_test)` → Uses the trained model to predict **house prices**.

### **Expected Output**
Mean Squared Error: 0.558  
R² Score: 0.575  
- The **MSE value of ~0.55** means the model makes moderate errors.  
- The **R² score of ~0.57** shows that the model explains **57% of the variance** in house prices.  

---

## **Exercise 2: Unsupervised Learning with K-Means Clustering**
### **Description**
This exercise implements **K-Means Clustering**, an unsupervised machine learning technique that groups similar data points into clusters.

### **Code Explanation**
1. **Synthetic Data Generation:**  
   - We generate 300 data points using `make_blobs()` with **4 distinct clusters**.
  
2. **K-Means Initialization:**  
   - `KMeans(n_clusters=4, random_state=42)` initializes the clustering algorithm with **4 clusters**.

3. **Training the Model:**  
   - `kmeans.fit(X)` assigns each data point to one of the 4 clusters.

4. **Cluster Labeling & Visualization:**  
   - `kmeans.labels_` stores the cluster assignments.
   - `kmeans.cluster_centers_` retrieves the **centroid positions**.
   - A scatter plot visualizes the clustering.

### **Why These Blanks Were Filled?**
- `n_clusters=4` → The dataset was generated with **4 clusters**, so we specify **4 clusters** for K-Means.
- `kmeans.fit(X)` → This method runs the **K-Means algorithm** and assigns each data point to a cluster.

### **Expected Output**
- The plot shows **four clusters**, with each point colored according to its cluster.  
- **Cluster centers** (centroids) are marked as red 'X' points.  

---

## **Exercise 3: Simple Neural Network for Regression**
### **Description**
This exercise implements a **feedforward neural network** using Keras for regression.

### **Code Explanation**
1. **Model Architecture:**  
   - The neural network consists of:
     - **Input Layer:** `Dense(64, activation='relu', input_shape=(num_features,))`
     - **Hidden Layer:** `Dense(64, activation='relu')`
     - **Output Layer:** `Dense(1, activation='linear')`, suitable for regression.

2. **Model Compilation:**  
   - Uses **Adam optimizer** with `learning_rate=0.001`, which adapts learning rates dynamically.
   - **MSE (Mean Squared Error)** is used as the loss function.

3. **Training the Model:**  
   - `model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)`
     - **50 epochs** allow the model to learn patterns.
     - **Batch size 32** balances performance and memory efficiency.

### **Why These Blanks Were Filled?**
- `input_shape=(num_features,)` → Ensures that the input layer correctly matches the feature count.
- `learning_rate=0.001` → A small learning rate prevents overshooting optimal weights.

### **Expected Output**
Epoch 1/50 - Loss: 2856.552  
Epoch 50/50 - Loss: 3.3242  
- The **loss decreases**, showing that the model is learning.  
- The final loss is **low**, meaning the model makes good predictions.  

---

## **Exercise 4: Updating the State in the GridWorld Environment**
### **Description**
This exercise implements state transitions in a **GridWorld** reinforcement learning environment.

### **Code Explanation**
1. **State Variables:**  
   - `self.state = (r, c)` stores the **agent's current position** in the grid.

2. **Action Handling:**  
   - If the action is **0 (up)**, the agent moves **one row up** (`r - 1`).
   - The `max(r - 1, 0)` ensures **the agent doesn’t move out of bounds**.

3. **Terminal State Check:**  
   - If the agent reaches a **terminal state**, a reward of **1** is returned.
   - Otherwise, the reward is **0**, and the game continues.

### **Why This Blank Was Filled?**
- `r = max(r - 1, 0)` → This ensures the agent moves **up by one row** but doesn’t exit the grid.

### **Expected Output**
- The agent correctly updates its **state** without moving out of bounds.  
- It reaches a **goal state** and receives a reward **1**.  

---

