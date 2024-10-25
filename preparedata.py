import numpy as np
from scipy.integrate import solve_bvp
import matplotlib.pyplot as plt
import pandas as pd

# Defining parameters
a1 = 0.01
M = 0.5
t = 0.01
x = np.pi / 4
a2 = 0.02
a3 = 0.03
E1 = 0.1
E2 = 0.2
E3 = 0.3
gamma1 = 0.9
gamma2 = 0.22
brickman = 0.3
a = -(1 + a1 * np.sin(np.pi * (x - t)) + a2 * np.sin(2 * np.pi * (x - t)) + a3 * np.sin(3 * np.pi * (x - t)))
b = (1 + a1 * np.sin(np.pi * (x - t)) + a2 * np.sin(2 * np.pi * (x - t)) + a3 * np.sin(3 * np.pi * (x - t)))
P = (-E1 * a1 * (np.pi ** 3) - E2 * a1 * (np.pi ** 3)) * np.cos(1 * np.pi * (x - t)) \
    + (-8 * E1 * a2 * (np.pi ** 3) - 8 * E2 * a2 * (np.pi ** 3)) * np.cos(2 * np.pi * (x - t)) \
    + (-27 * E1 * a3 * (np.pi ** 3) - 27 * E2 * a3 * (np.pi ** 3)) * np.cos(3 * np.pi * (x - t)) \
    + (E3 * a1 * (np.pi ** 2)) * np.sin(1 * np.pi * (x - t)) \
    + (4 * E3 * a2 * (np.pi ** 2)) * np.sin(2 * np.pi * (x - t)) \
    + (9 * E3 * a3 * (np.pi ** 2)) * np.sin(3 * np.pi * (x - t))

# Differential equations
def mat4ode(x, y):
    cc = ((M ** 2) * (y[0] + 1) + P) / (gamma1 + 3 * (y[1] ** 2) * gamma2)
    dd = -brickman * y[1] * (gamma1 * y[1] + gamma2 * (y[1] ** 3))
    return np.vstack((y[1], cc, y[3], dd))

# Boundary conditions
def mat4bc(ya, yb):
    return np.array([ya[0], yb[0], ya[2], yb[2] - 1])

# Initial guess for the solution
def mat4init(x):
    return np.zeros((4, x.size))

# Setting up the problem with range from -1 to 1
x = np.linspace(-1, 1, 1000)
solinit = mat4init(x)

# Solving the BVP
sol = solve_bvp(mat4ode, mat4bc, x, solinit)

# Extracting the solution
eta = sol.x
f = sol.y

# Saving the data to a single CSV file
# Creating a DataFrame for both plots (f[2, :] and f[0, :]) including all parameters
df = pd.DataFrame({
    'a1': a1,
    'M': M,
    't': t,
    'x': x,
    'a2': a2,
    'a3': a3,
    'E1': E1,
    'E2': E2,
    'E3': E3,
    'gamma1': gamma1,
    'gamma2': gamma2,
    'brickman': brickman,
    'a': a,
    'b': b,
    'P': P,
    'eta': eta,
    "Temp": f[2, :],
    'velocity': f[0, :]
})
df.to_csv('f_vs_eta.csv', index=False)

# Plotting the 1st derivative of the stream function against eta and saving it to a file
# Plotting the first plot for f[2, :]
plt.figure()  # Create a new figure
plt.plot(eta, f[2, :])
plt.xlabel('eta')
plt.ylabel("f'(eta)")
plt.title('1st Derivative of Stream Function Against eta')
plt.grid()
plt.savefig('f2_vs_eta.png')  # Save the plot as a PNG file
plt.close()  # Close the current figure

# Plotting the second plot for f[0, :]
plt.figure()  # Create a new figure
plt.plot(eta, f[0, :])
plt.xlabel('eta')
plt.ylabel('f(eta)')
plt.title('Stream Function f Against eta')
plt.grid()
plt.savefig('f0_vs_eta.png')  # Save the plot as a PNG file
plt.close()  # Close the current figure
