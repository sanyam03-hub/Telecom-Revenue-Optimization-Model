import sys
import os
sys.path.append('.')

from src.data_generation import TelecomDataGenerator

# Generate synthetic dataset
print("Generating synthetic dataset...")
generator = TelecomDataGenerator(n_customers=5000)
datasets = generator.generate_complete_dataset()

# Create directory if it doesn't exist
os.makedirs('data/raw', exist_ok=True)

# Save the master dataset
df = datasets['master_dataset']
df.to_csv('data/raw/master_dataset.csv', index=False)
print("Data generated successfully!")
print(f"Dataset shape: {df.shape}")