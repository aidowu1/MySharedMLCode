'''
Basic Tutorials on using Tensor Flow
(c) Ade Idowu 04Jan2018
'''

'''
This is a simple demo of Tensorflow.
It is going to demonstrate how to compute the minium point of a 2D Rosenbrock function.
This function is defined as:

rosen(x) = .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2

or more parametrically as:

rosen(x; c) = c[0][0]*(c[1][0] + c[2][0]*x[0][0])**2 + (c[3][0]*x[1][0] + c[4][0]*x[0][0])**2

Optimal values of this fuction are x = [[1.0], [1.0]]
'''

print("Starting demo...")

import numpy as np
import tensorflow as tf

# Define minimum variable x
x = tf.Variable(tf.zeros([2, 1], dtype=tf.float32))

# Define the Rosenbrock function coefficeints as placeholder c
c = tf.placeholder(tf.float32, [5, 1])

# Define the Rosenbrock function (symbolically) we are trying to minimize
rosen = c[0][0]*(c[1][0] + c[2][0]*x[0][0])**2 + (c[3][0]*x[1][0] + c[4][0]*x[0][0])**2

# Define the Tensorflow optimizer, we are going to use the Gradient Descent find the optimal (minimum) point
# of this function (using a learning rate of 0.01)
learn_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(rosen)

# Define the coefficient c
coefficients = np.array([[0.5], [1.0], [-1.0], [1.0], [-1.0]])

# Initialize Tensorflow
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print("Initial values of x are: {}".format(session.run(x)))

# Now let us run the 1st iteration of this computation:
session.run(optimizer, feed_dict={c:coefficients})
print("1st iteration value of x are: \n{}\n".format(session.run(x)))


# Now let us see the results of this computation for many iterations
max_iter = 2000
for i in range(max_iter):
    session.run(optimizer, feed_dict={c:coefficients})
    if i%200 == 0:
        print("{0}th iteration value of x are: \n{1}\n".format(i, session.run(x)))






