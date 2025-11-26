Toy Classification using Machine Learning -- MITx 6.86x coursework

In this repository, I will demonstrate my understanding of perceptron algorithm, average perceptron algorithm, and pegasos algorthm by coding each algorithm's parameter update machanism and finding optimal theta and theta_0.

Perceptron algorihtm for Binary classification:
  <img width="555" height="184" alt="image" src="https://github.com/user-attachments/assets/57257123-a73c-491d-a422-5555aafafded" />
  Upgrads only mistakes:
  <img width="975" height="459" alt="image" src="https://github.com/user-attachments/assets/7965452d-a242-4abb-bd49-b7560ce46eea" />

Avergae Perceptron algorihtm for Binary classification:
  Identical algrithm, only differs when updating:
  Record every single trainging set's theta and theta_0, finally average them out.
  <img width="343" height="98" alt="image" src="https://github.com/user-attachments/assets/d8e77745-98b2-44a0-a944-11057f0f8eb8" />
  So perceprton optimal values are the final updates, while avg perceptron optinal values are the average results of all theta, including the case when algorithm correctly classified the result and theta not changed for ith trainging set.

Pegasos algorthm for Binary classification:
  It is equavilent to applying stochastic gradient descent to SVM algorithm.
  <img width="865" height="440" alt="image" src="https://github.com/user-attachments/assets/a42e7759-59fd-4a6f-8523-fc7f7acda853" />
  The algorithm/objective function contian:
    a hinge loss function which will be great than 0 as long as traing set data points outside the marginal boundray.
    <img width="975" height="246" alt="image" src="https://github.com/user-attachments/assets/8de9b84d-6425-458e-b0bc-cc917384bf69" />
    Plus a regulzation term which determin how much distance is between the decision and marginal boundry, the larger the boundry, the more generalized is the algorithm.
    <img width="975" height="354" alt="image" src="https://github.com/user-attachments/assets/382f0bfa-e6d8-41f2-a613-d79aa9c3e736" />

  Hence the updating:
  <img width="865" height="440" alt="image" src="https://github.com/user-attachments/assets/cf0b9ce1-8d4b-4393-81ea-88b05484b41a" />
  Given positive regulization term, theta and tehta_0 would expected to be smaller than the other two algorithm as minimizing objective function with larger lambdait will put more weight on generalziation and penalized lambda being too large, which in term lead 1/||theta|| become large, and hence larger margin.

  
Workflow:
  explore data
  finding best theta and theta_0
  plot graph
  test results



