import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Given data
X = np.array([-5.25, -6.13, -7.13, -6.75, -6.88, -7.38, -6.77, -7.38, -6.2, -6.84,
              -8.03, -6.5, -7.01, -7.65, -6.94, -6.28, -6.4, -6.89, -6.8, -5.7])
Y = np.array([-5.23, -6.36, -5.9, -6.46, -6.46, -5.74, -6.31, -5.99, -5.97, -6.36,
              -6.98, -5.69, -5.74, -6.31, -7.8, -4.96, -6.31, -6.0, -6.56, -5.74])

# Calculate linear regression
slope, intercept, _, _, _ = linregress(X, Y)
regression_line = lambda x: slope * x + intercept

print("Slope: ",slope, "Intercept: ",intercept)

# Scatter plot with regression line
plt.figure(figsize=(10, 10))
plt.scatter(X, Y, color='b', alpha=0.7, label='Data Points')
plt.plot(X, regression_line(X), color='r', linestyle='--', label='Regression Line')
plt.axhline(np.mean(Y), color='r', linestyle='--', label='Mean Y')
plt.axvline(np.mean(X), color='g', linestyle='--', label='Mean X')
plt.xlabel('X Coordinates (Economic View)')
plt.ylabel('Y Coordinates (Social View)')
plt.title('Political Compass Data with Regression Line')
plt.legend()
plt.grid(True)
plt.show()



# (Optional) Hypothesis Testing (replace with your specific test)
from scipy import stats
# # Example: Test if there's a statistically significant difference in means
tstat, pval = stats.ttest_ind(X, Y)
if pval < 0.05:
    print("There is a statistically significant difference between X and Y means.")
else:
    print("There is no statistically significant difference between X and Y means.")