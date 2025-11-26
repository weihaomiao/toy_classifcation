# Toy Classification using Machine Learning — MITx 6.86x Coursework

This repository demonstrates implementations of the **Perceptron**, **Average Perceptron**, and **Pegasos** algorithms for binary classification. Each algorithm is coded from scratch, including parameter updates and the search for optimal \(\theta\) and \(\theta_0\).

---

## 1. Algorithms

### 1.1 Perceptron
- Standard binary classifier.
- Updates only on misclassified points:
- Algorithm:  
  ![Perceptron Decision Boundary](https://github.com/user-attachments/assets/57257123-a73c-491d-a422-5555aafafded)
- Updates occur only on mistakes:  
  ![Perceptron Updates](https://github.com/user-attachments/assets/7965452d-a242-4abb-bd49-b7560ce46eea)
- Decision boundary with toy data:
  
  ![Decision Boundary — Perceptron](https://github.com/user-attachments/assets/d2b3fdb5-6ae9-454d-bb74-3a40f473c0d5)
---

### 1.2 Average Perceptron
- Algorithm same as perceptron
- Key difference: **averages all theta and theta_0 values across training steps**, including unchanged ones.
- Updates recorded for every training sample:  
  ![Average Perceptron](https://github.com/user-attachments/assets/d8e77745-98b2-44a0-a944-11057f0f8eb8)
- Decision boundary with toy data:

  <img width="640" height="480" alt="Figure_2" src="https://github.com/user-attachments/assets/937e9162-dd6d-4796-b385-e43818024c2e" />

---

### 1.3 Pegasos (SGD for SVM)
- Uses hinge loss and regularization term for binary classification.
- Algorithm:
	<img width="975" height="354" alt="image" src="https://github.com/user-attachments/assets/ef773239-0263-4089-9a33-f6485535c109" />
- Updates based on randomly selected n individual samples:

  ![Sample Updates](https://github.com/user-attachments/assets/dd2c8d9d-cccc-4b0e-aa1c-fcf18e83e806)

- Decision boundary with toy data:

  ![Decision Boundary — Perceptron](https://github.com/user-attachments/assets/0370f16e-16a0-421c-8800-8f6a152a1893)

---

## 2. Workflow

1. **Data Exploration** – Data are given
2. **Parameter Optimization** – Find best \(\theta\) and \(\theta_0\) for each algorithm.
3. **Visualization** – Plot decision boundaries and updates; clearly distinguish lines for Perceptron, Average Perceptron, and Pegasos.
4. **Test** – Run test to see if algorithm haven applied correctly.



