Toy Classification using Machine Learning -- MITx 6.86x coursework

In this repository, I will demonstrate my understanding of perceptron algorithm、average perceptron algorithm, and pegasos algorthm by coding each algorithm and finding their optimal theta and theta_0.

Perceptron algorihtm for Binary classification:
  <img width="555" height="184" alt="image" src="https://github.com/user-attachments/assets/57257123-a73c-491d-a422-5555aafafded" />
  Upgrads only mistakes:
  <img width="975" height="459" alt="image" src="https://github.com/user-attachments/assets/7965452d-a242-4abb-bd49-b7560ce46eea" />

Avergae Perceptron algorihtm for Binary classification:
  Identical algrithm, only differs when updating:
  Record every single trainging set's theta and theta_0, finally average them out.
  <img width="343" height="98" alt="image" src="https://github.com/user-attachments/assets/d8e77745-98b2-44a0-a944-11057f0f8eb8" />
  So perceprton optimal values are the final updates, while avg perceptron optinal values are the average results of all theta, including the case when algorithm correctly classified the result and theta not changed for ith trainging set.


Workflow:
  explore data
  finding best theta and theta_0
  plot graph
  test results


