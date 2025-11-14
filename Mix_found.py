import pandas as pd
import os

# Define the file path
file_path = ""

# Check if file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit(1)

# Read the file using pandas
df = pd.read_csv(file_path, delimiter='\t')

# Sort by Acc@1 in descending order and get top 8
top_acc1 = df.sort_values(by='Acc@1', ascending=False).head(8)

# Sort by Acc@5 in descending order and get top 8
top_acc5 = df.sort_values(by='Acc@5', ascending=False).head(8)

# Print the results
print("Top 8 values for Acc@1:")
print("=" * 50)
for i, (timestamp, acc1, acc5) in enumerate(zip(top_acc1['Timestamp'], top_acc1['Acc@1'], top_acc1['Acc@5']), 1):
    print(f"{i}. {acc1:.3f} (Timestamp: {timestamp})")

print("\nTop 8 values for Acc@5:")
print("=" * 50)
for i, (timestamp, acc1, acc5) in enumerate(zip(top_acc5['Timestamp'], top_acc5['Acc@1'], top_acc5['Acc@5']), 1):
    print(f"{i}. {acc5:.3f} (Timestamp: {timestamp})")
