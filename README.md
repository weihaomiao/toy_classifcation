Toy Classification using Machine Learning — MITx 6.86x Coursework

This repository demonstrates implementations of the Perceptron, Average Perceptron, and Pegasos algorithms for binary classification. Each algorithm is coded from scratch, including parameter updates and the search for optimal 
𝜃
θ and 
𝜃
0
θ
0
	​

.

1. Algorithms
1.1 Perceptron

Standard binary classifier.

Updates only on misclassified points:

If 
𝑦
𝑖
(
𝜃
⋅
𝑥
𝑖
+
𝜃
0
)
≤
0
:
𝜃
←
𝜃
+
𝑦
𝑖
𝑥
𝑖
,
𝜃
0
←
𝜃
0
+
𝑦
𝑖
If y
i
	​

(θ⋅x
i
	​

+θ
0
	​

)≤0:θ←θ+y
i
	​

x
i
	​

,θ
0
	​

←θ
0
	​

+y
i
	​


Visuals:

Decision boundary with toy data:


Updates occur only on mistakes:


1.2 Average Perceptron

Identical update rule as Perceptron.

Key difference: averages all 
𝜃
θ and 
𝜃
0
θ
0
	​

 values across training steps, including unchanged ones.

𝜃
ˉ
=
1
𝑁
∑
𝑡
=
1
𝑁
𝜃
(
𝑡
)
,
𝜃
ˉ
0
=
1
𝑁
∑
𝑡
=
1
𝑁
𝜃
0
(
𝑡
)
θ
ˉ
=
N
1
	​

t=1
∑
N
	​

θ
(t)
,
θ
ˉ
0
	​

=
N
1
	​

t=1
∑
N
	​

θ
0
(t)
	​


Visuals:

Updates recorded for every training sample:


1.3 Pegasos (Stochastic Gradient Descent for SVM)

Uses hinge loss and regularization for binary classification.

Update rule:

𝜃
←
(
1
−
𝜂
𝜆
)
𝜃
+
𝜂
𝑦
𝑖
𝑥
𝑖
if margin violated
,
𝜃
0
←
𝜃
0
+
𝜂
𝑦
𝑖
θ←(1−ηλ)θ+ηy
i
	​

x
i
	​

if margin violated,θ
0
	​

←θ
0
	​

+ηy
i
	​


Objective function:

min
⁡
𝜃
,
𝜃
0
𝜆
2
∥
𝜃
∥
2
+
1
𝑛
∑
𝑖
=
1
𝑛
max
⁡
(
0
,
1
−
𝑦
𝑖
(
𝜃
⋅
𝑥
𝑖
+
𝜃
0
)
)
θ,θ
0
	​

min
	​

2
λ
	​

∥θ∥
2
+
n
1
	​

i=1
∑
n
	​

max(0,1−y
i
	​

(θ⋅x
i
	​

+θ
0
	​

))

Regularization term (
𝜆
λ) controls the margin and generalization.

Visuals:

Decision boundary with Pegasos:


Hinge loss illustration:


Effect of regularization:


Update rule visualization:


Tip: Use distinct line styles, colors, or markers for each algorithm in plots to make comparisons clear.

2. Workflow

Data Exploration – Visualize toy datasets and understand feature distribution.

Parameter Optimization – Find best 
𝜃
θ and 
𝜃
0
θ
0
	​

 for each algorithm.

Visualization – Plot decision boundaries and updates; clearly distinguish lines for Perceptron, Average Perceptron, and Pegasos.

Evaluation – Compare classification accuracy and behavior across algorithms.


