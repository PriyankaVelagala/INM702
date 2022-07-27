# INM702: Mathematics and Programming for AI 
Priyanka Velagala 

# Task 1: Statistical Analysis of Shortest Path Algorithms 
For full report, see: [Task1-Report.pdf](https://github.com/PriyankaVelagala/Mathematics-and-Programming-for-AI/blob/main/Task1/Task1-Report.pdf)

You need to develop a simple game consisting of a rectangular grid (of size height x width) where each cell has a random value between 0 and n. An agent starts at the upper-left corner of the grid and must reach the lower-right corner of the grid as fast as possible. Accordingly, the task consists on finding the shortest path.

There are two game modes:
- The time spent on a cell is the number on this cell
- The time spent on a cell is the absolute of the difference between the previous cell the agent was on and the current cell it is on

Implement the game in a structured and flexible way to allow the selection of game modes and parameters. Develop your own heuristic algorithm. Identify simple criteria and strategies to find short paths. This algorithm should be taken as a baseline. Implement the Dijkstra's algorithm to find the shortest path between two points. Plan and implement a statistical analysis to characterize the length of the shortest path in dependence of several parameters of the grid, and comparing the two game modes.

# Task 2: Analysis of Factors affecting Linear Regression 
For full report see: [Task2-Report.pdf](https://github.com/PriyankaVelagala/Mathematics-and-Programming-for-AI/blob/main/Task2/Task2-Report.pdf)
### Outliers 
The presence of outliers in a data model tend to affect the accuracy with the model parameters
are estimated. During analysis an increasing percentage of outliers were introduced to the dataset
and metrics such as the estimated slope in relation to the true value and the standard deviation
of the estimated slope were tracked. It was observed that both measures increase linearly as the
percentage of outliers increases. A similar observation is made in relation to RMSE which implies
the larger the number of outliers in a dataset the worse the model is in predicting the true value.

### Multi-collinearity 
Multicollinearity is a condition when two or more variables in a multiple regression model are
highly correlated. This violates one of the assumptions in linear regression where the model
assumes all covariates are independent. A consequence of having multicollinearity in regression
models is poor estimation of model parameters. We note that covariate coefficients become sensitive
to small changes in the model and can swing wildly based on which variables are in the model. 

# Task 3: Implementing a Neural Network using NumPy 
For full report, see: [Task3,4-Report.pdf](https://github.com/PriyankaVelagala/Mathematics-and-Programming-for-AI/blob/main/Task3/Task3%2C4-Report.pdf)

In this task, we implement a multi-layer neural network using Python’s NumPy library. The neural network will then be configured to train against a multiclass classification problem. For the purposes of this task, we’ll be training it on the Fashion-MNIST dataset. This dataset consists of 70,000 28x28 grayscale images and contains a total of 10 classes.The neural network class was implemented such that it accepts a customizable set of parameters for number of hidden layers, number of nodes and the activation functions. Some of these network parameters are discussed in greater detail under Task 4. Here, we will focus on the implemented activation functions and optimizers.

# Task 4: Implementing Neural Network using PyTorch
For full report, see: [Task3,4-Report.pdf](https://github.com/PriyankaVelagala/Mathematics-and-Programming-for-AI/blob/main/Task4/Task3%2C4-Report.pdf)

For this task the CIFAR-10 dataset was used. The dataset consists of a total of 60,000 images where each image is labelled with one of the following classes – airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

The objective of this task is to implement a classifier using the PyTorch library to accurately label the images in the CIFAR-10 dataset. For this multiclass classification problem, the strategy to find a suitable network architecture will be by first establishing a baseline neural network model with reasonable accuracy (approx. 50%) and then tuning the model parameters and network architecture through techniques taught in the course.
