import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
regs = [0] + [10**i for i in range(-8, 1)]
# print(len(regs))
train_error = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.55]

# Create a Seaborn line plot
sns.lineplot(x=regs, y=train_error)

# Set the x-axis to log scale to display the range of regularization values properly
plt.xscale('symlog', linthresh=1e-8)  # Use 'symlog' because your data includes 0

# Add labels and title
plt.xlabel('Regularization (log scale)')
plt.ylabel('Train Error')
plt.title('Train Error by Regularization')

# Show the plot
plt.show()
