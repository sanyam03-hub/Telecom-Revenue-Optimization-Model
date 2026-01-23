import unittest
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generation import TelecomDataGenerator

class TestDataGeneration(unittest.TestCase):
    """Test cases for synthetic data generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = TelecomDataGenerator(n_customers=100)
        
    def test_demographics_shape(self):
        """Test if demographics dataframe has correct shape."""
        df = self.generator.generate_customer_demographics()
        self.assertEqual(len(df), 100)
        self.assertTrue('customer_id' in df.columns)
        self.assertTrue('age' in df.columns)
        
    def test_data_integrity(self):
        """Test if generated data respects constraints."""
        df = self.generator.generate_customer_demographics()
        self.assertTrue(df['age'].min() >= 18)
        self.assertTrue(df['age'].max() <= 80)
        
    def test_full_pipeline(self):
        """Test the complete generation pipeline."""
        datasets = self.generator.generate_complete_dataset()
        self.assertTrue('master_dataset' in datasets)
        self.assertEqual(len(datasets['master_dataset']), 100)
        
if __name__ == '__main__':
    unittest.main()
