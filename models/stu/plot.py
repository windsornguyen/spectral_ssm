import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Load and prepare the data
df = pd.read_csv('stu_usage_data.csv', index_col=0)
df.index.name = 'Forward Pass'

# Calculate average percentages
avg_percentages = df.mean()

# Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
fig.suptitle('STU Usage Analysis', fontsize=16, fontweight='bold')

# Plot 1: Bar Plot of Average STU Usage Percentages
sns.barplot(x=avg_percentages.index, y=avg_percentages.values, ax=ax1)
ax1.set_title('Average STU Usage Percentages', fontsize=14)
ax1.set_xlabel('STU', fontsize=12)
ax1.set_ylabel('Average Usage Percentage', fontsize=12)
ax1.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))

# Add value labels on top of each bar
for i, v in enumerate(avg_percentages):
    ax1.text(i, v, f'{v:.2f}%', ha='center', va='bottom')

# Plot 2: Line Plot of Individual STU Usage
for column in df.columns:
    sns.regplot(x=df.index, y=column, data=df, ax=ax2, scatter=False, label=column)

ax2.set_title('Individual STU Usage Trends', fontsize=14)
ax2.set_xlabel('Forward Passes', fontsize=12)
ax2.set_ylabel('Usage Percentage', fontsize=12)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig('stu_usage_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Plot saved as 'stu_usage_analysis.png'")

# Generate summary statistics
summary = df.describe()
summary.to_csv('stu_usage_summary.csv')
print("Summary statistics saved as 'stu_usage_summary.csv'")

# Identify the most and least used STUs
most_used = df.mean().idxmax()
least_used = df.mean().idxmin()
print(f"\nMost frequently used: {most_used}")
print(f"Least frequently used: {least_used}")

# Print the average usage percentages
print("\nAverage Usage Percentages:")
for stu, percentage in avg_percentages.items():
    print(f"{stu}: {percentage:.2f}%")
