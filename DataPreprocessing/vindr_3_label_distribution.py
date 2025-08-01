import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = '/Users/admin/Working/Data/vinbigdata-chest-xray-30-percent/train.csv'
df = pd.read_csv(CSV_PATH)
print(f"Total rows loaded: {len(df)}")

# Define condition columns (excluding 'image_id' and the last column)
condition_columns = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Pleural Effusion', 'Pneumothorax']

# Calculate the sum of each condition (number of positive cases)
label_distribution = df[condition_columns].sum()
total_samples = len(df)

# Calculate percentage distribution
percent_distribution = (label_distribution / total_samples) * 100

plt.figure(figsize=(10, 6))
percent_distribution.plot(kind='bar')
plt.title('Label Distribution in Dataset')
plt.xlabel('Conditions')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(percent_distribution):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.show()

print("\nLabel Distribution (Count):")
print(label_distribution)
print("\nLabel Distribution (Percentage):")
print(percent_distribution.round(2))