# Bonus Assignment  - CS4700 Artificial Intelligence

## Student Information
- **Name:** Niharika Goud Cika
- **Student ID:** 700773687
- **Course:** CS4700 - Artificial Intelligence
- **Semester:** Spring 2025
- **University:** University of Central Missouri  

## Overview
This repository contains solutions for **Bonus Assignment** in **CS4700 - Artificial Intelligence**. The assignment covers different AI and ML techniques including **Linear Regression, K-Means Clustering, Neural Networks, and Reinforcement Learning**.

The following exercises were implemented:
- **Exercise 1:** House Price Prediction using Linear Regression
- **Exercise 2:** K-Means Clustering
- **Exercise 3:** Simple Neural Network for Regression
- **Exercise 4:** Updating the State in the GridWorld Environment  


## **Exercise 1: House Price Prediction using Linear Regression**
### **Description**
This exercise uses **Linear Regression** to predict house prices based on different features like median income, house age, and location.

### **Code Explanation**
The code first loads the **California Housing dataset** and separates it into input features and target prices. It then splits the data into **80% training and 20% testing**, ensuring the model learns from most of the data while keeping some for evaluation. A **Linear Regression model** is created and trained using the training data. Once trained, the model makes predictions on the test set. To check its performance, two metrics are used: **Mean Squared Error (MSE)**, which shows how far the predictions are from actual values, and **R² Score**, which tells how well the model explains the house price variations. A lower **MSE** and a higher **R² score** mean better predictions.

### **Why These Blanks Were Filled?**
- `test_size=0.2` → Splitting data into **80% training and 20% testing** is a common practice.  
- `model.fit(X_train, y_train)` → Trains the **Linear Regression** model using training data.  
- `model.predict(X_test)` → Uses the trained model to predict house prices on test data.  

### **Expected Output**
Mean Squared Error: 0.558  
R² Score: 0.575  
- The **MSE value of ~0.55** means the model makes moderate errors.  
- The **R² score of ~0.57** shows that the model explains **57% of the variance** in house prices.  

---

## **Exercise 2: Unsupervised Learning with K-Means Clustering**
### **Description**
This exercise applies **K-Means Clustering**, a technique that groups data points into clusters based on similarities.

### **Code Explanation**
The code creates a dataset with **300 data points** divided into **4 groups (clusters)** using `make_blobs()`. Then, a **K-Means model** is created and set to find **4 clusters**. The model is trained on the dataset, meaning it assigns each point to one of the **4 clusters** based on distance from cluster centers. After training, each data point is labeled with its respective cluster, and the **cluster centers** are found. The results are visualized using a scatter plot, where each point is colored based on its cluster, and red **‘X’ markers** represent the cluster centers.

### **Why These Blanks Were Filled?**
- `n_clusters=4` → The dataset was created with **4 clusters**, so we set **4 clusters** in K-Means.  
- `kmeans.fit(X)` → Runs the **K-Means algorithm** and assigns each data point to a cluster.  

### **Expected Output**
- A scatter plot showing **four different clusters** with different colors.  
- Red 'X' markers representing **cluster centers**.  

---

## **Exercise 3: Simple Neural Network for Regression**
### **Description**
This exercise builds a **neural network** to predict continuous values using **Keras**.

### **Code Explanation**
The code first defines a **neural network model** using the **Sequential API** in Keras. The model has **three layers**: an **input layer** that takes in the feature data, a **hidden layer** that learns patterns using **ReLU activation**, and an **output layer** that gives the final prediction using a **linear activation function**. The model is compiled using the **Adam optimizer** (`learning_rate=0.001`), which helps adjust learning dynamically, and **Mean Squared Error (MSE)** is chosen as the loss function because this is a regression problem. The model is trained for **50 epochs** with a **batch size of 32**, meaning it learns in small groups rather than processing all data at once. Over time, the loss decreases, showing that the model is improving.

### **Why These Blanks Were Filled?**
- `input_shape=(num_features,)` → Ensures the input layer matches the number of features in the dataset.  
- `learning_rate=0.001` → A small learning rate prevents the model from making large jumps and missing the best solution.  

### **Expected Output**
Epoch 1/50 - Loss: 2866.5252  
Epoch 50/50 - Loss: 0.7012 
- The **loss decreases**, showing the model is learning and improving its predictions.  

---

## **Exercise 4: Updating the State in the GridWorld Environment**
### **Description**
This exercise updates the **position of an agent** in a **GridWorld** environment when it moves.

### **Code Explanation**
The function `step(action)` updates the **agent’s position** based on the action taken. The **state** of the agent is stored as `(row, column)`, and if the action is **"up" (action 0)**, the row index decreases. The function ensures that the agent does not move **out of bounds** by using `max(r - 1, 0)`, which keeps it from going below **row 0**. The agent keeps moving until it reaches a **terminal state**, where it receives a reward of **1**. If the agent is not in a terminal state, the function returns **0** as a reward.

### **Why This Blank Was Filled?**
- `r = max(r - 1, 0)` → Ensures that the agent moves **one step up** but does not go beyond the top boundary.  

### **Expected Output**
- The agent correctly updates its **state** without moving out of bounds.  
- It reaches a **goal state** and receives a reward **1**.  
