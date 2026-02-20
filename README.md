# Linear Regression from Scratch 🚀

A from-scratch implementation of the Linear Regression algorithm using **Gradient Descent** and pure **NumPy**. This project demonstrates the core mathematical concepts behind machine learning without relying on high-level libraries like `scikit-learn`.

## 🛠 Features
* Built entirely with Python and NumPy.
* Matrix operations (Vectorization) for optimized performance.
* Clean and readable code architecture (OOP approach).

## 💻 Prerequisites
Make sure you have Python installed. Then, install the required dependencies:

```bash
pip install numpy pandas matplotlib

```

## 🚀 Quick Start

1. Clone this repository:

```bash
git clone [https://github.com/Mango686743/LinearRegression_Project.git](https://github.com/Mango686743/LinearRegression_Project.git)

```

2. Run the main script to train the model and see predictions:

```bash
python main.py

```

## 🧠 Under the Hood (Math)

The model uses the Mean Squared Error (MSE) cost function and updates weights using the Gradient Descent rule:

* `dw = (1/m) * X.T.dot(y_pred - y)`
* `db = (1/m) * sum(y_pred - y)`


## 🔮 Future Work: Logistic Regression (My Eureka Moment)

**The Flaw of Linear Regression:** During testing, I realized Linear Regression predicts bounded metrics (like a 0-100 test score) linearly into infinity ($-\infty$ to $+\infty$). A student studying 1000 hours would get an impossible score of 5000. 

**The Mathematical Solution (Logistic Growth):**
Instead of using a hard cutoff like `np.clip()`, the model should mimic natural bounded growth (Differential Equations from Calculus 1). We need to pass the linear equation through a **Sigmoid function** to squeeze all predictions into a 0 to 1 range (which can then be scaled to 0-100).

**Blueprint for Implementation:**
1. **Feed-forward (predict):** - Calculate the linear step: $z = X \cdot w + b$
   - Apply Sigmoid: $\hat{y} = \frac{1}{1 + e^{-z}}$
2. **Backpropagation (fit):** - Thanks to the magical math derivative of Log-Loss (Binary Cross-Entropy) paired with Sigmoid, the gradients $dw$ and $db$ formulas remain **EXACTLY THE SAME** as Linear Regression!
   - Just use the new Sigmoid $\hat{y}$ in the existing gradient code.