"""
Synthetic Telecom Customer Data Generation

This module generates realistic synthetic telecom customer data for the revenue optimization project.
Includes customer demographics, usage patterns, billing info, and campaign data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import random
from typing import Tuple, Dict, List

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class TelecomDataGenerator:
    """Generates synthetic telecom customer data for analytics and ML modeling."""
    
    def __init__(self, n_customers: int = 10000):
        """
        Initialize data generator.
        
        Args:
            n_customers: Number of customers to generate
        """
        self.n_customers = n_customers
        self.start_date = datetime(2022, 1, 1)
        self.end_date = datetime(2024, 9, 30)
        
        # Define plan types and their characteristics
        self.plan_types = {
            'Basic': {'price': 25, 'data_gb': 2, 'minutes': 500},
            'Standard': {'price': 45, 'data_gb': 10, 'minutes': 1000},
            'Premium': {'price': 75, 'data_gb': 50, 'minutes': 2000},
            'Unlimited': {'price': 95, 'data_gb': 999, 'minutes': 9999}
        }
        
        # Define cities and their characteristics
        self.cities = {
            'New York': {'population': 8000000, 'income_multiplier': 1.3},
            'Los Angeles': {'population': 4000000, 'income_multiplier': 1.2},
            'Chicago': {'population': 2700000, 'income_multiplier': 1.1},
            'Houston': {'population': 2300000, 'income_multiplier': 1.0},
            'Phoenix': {'population': 1600000, 'income_multiplier': 0.95},
            'Philadelphia': {'population': 1500000, 'income_multiplier': 1.05},
            'San Antonio': {'population': 1500000, 'income_multiplier': 0.9},
            'San Diego': {'population': 1400000, 'income_multiplier': 1.15},
            'Dallas': {'population': 1300000, 'income_multiplier': 1.0},
            'San Jose': {'population': 1000000, 'income_multiplier': 1.4}
        }
        
    def _generate_hashed_id(self, customer_idx: int) -> str:
        """Generate privacy-compliant hashed customer ID."""
        raw_id = f"customer_{customer_idx}_{np.random.randint(1000, 9999)}"
        return hashlib.sha256(raw_id.encode()).hexdigest()[:16]
    
    def generate_customer_demographics(self) -> pd.DataFrame:
        """Generate customer demographic data."""
        print("Generating customer demographics...")
        
        customers = []
        
        for i in range(self.n_customers):
            # Generate customer ID
            customer_id = self._generate_hashed_id(i)
            
            # Demographics
            age = np.random.normal(42, 15)
            age = max(18, min(80, int(age)))  # Clamp between 18-80
            
            gender = np.random.choice(['Male', 'Female', 'Other'], p=[0.48, 0.48, 0.04])
            
            # Location (weighted by population)
            city_weights = [city['population'] for city in self.cities.values()]
            city_weights = np.array(city_weights) / sum(city_weights)
            city = np.random.choice(list(self.cities.keys()), p=city_weights)
            
            # Income based on age, location, and some randomness
            base_income = 30000 + (age - 18) * 800 + np.random.normal(0, 15000)
            income = max(20000, base_income * self.cities[city]['income_multiplier'])
            
            # Tenure (how long they've been a customer)
            tenure_months = np.random.exponential(24)  # Average 2 years
            tenure_months = min(60, max(1, int(tenure_months)))  # Cap at 5 years
            
            # Join date
            days_back = tenure_months * 30
            join_date = self.end_date - timedelta(days=int(days_back))
            
            customers.append({
                'customer_id': customer_id,
                'age': age,
                'gender': gender,
                'city': city,
                'annual_income': int(income),
                'tenure_months': tenure_months,
                'join_date': join_date.strftime('%Y-%m-%d')
            })
        
        return pd.DataFrame(customers)
    
    def generate_usage_and_billing(self, demographics_df: pd.DataFrame) -> pd.DataFrame:
        """Generate usage patterns and billing information."""
        print("Generating usage and billing data...")
        
        usage_billing = []
        
        for _, customer in demographics_df.iterrows():
            # Plan selection based on income and age
            if customer['annual_income'] > 80000 and customer['age'] < 45:
                plan_probs = [0.1, 0.2, 0.4, 0.3]  # Bias toward premium plans
            elif customer['annual_income'] > 50000:
                plan_probs = [0.15, 0.35, 0.35, 0.15]  # Balanced
            else:
                plan_probs = [0.4, 0.4, 0.15, 0.05]  # Bias toward basic plans
            
            plan_type = np.random.choice(list(self.plan_types.keys()), p=plan_probs)
            plan_info = self.plan_types[plan_type]
            
            # Usage patterns based on plan and demographics
            base_data_usage = plan_info['data_gb'] * np.random.uniform(0.3, 1.2)
            if customer['age'] < 30:
                base_data_usage *= 1.5  # Young people use more data
            elif customer['age'] > 60:
                base_data_usage *= 0.7  # Older people use less data
                
            base_minutes = plan_info['minutes'] * np.random.uniform(0.2, 0.9)
            
            # Monthly variation
            monthly_data_usage = max(0, np.random.normal(base_data_usage, base_data_usage * 0.3))
            monthly_minutes = max(0, np.random.normal(base_minutes, base_minutes * 0.25))
            
            # OTT/Media app usage (streaming services)
            ott_usage_hours = np.random.exponential(15) if customer['age'] < 50 else np.random.exponential(8)
            ott_usage_hours = min(100, ott_usage_hours)  # Cap at 100 hours/month
            
            # Calculate ARPU with some variation
            base_arpu = plan_info['price']
            # Add overage charges
            if monthly_data_usage > plan_info['data_gb']:
                overage_gb = monthly_data_usage - plan_info['data_gb']
                overage_charge = overage_gb * 10  # $10 per GB overage
                base_arpu += overage_charge
            
            # Add some randomness for additional services
            additional_services = np.random.normal(5, 10)
            arpu = max(base_arpu, base_arpu + additional_services)
            
            usage_billing.append({
                'customer_id': customer['customer_id'],
                'plan_type': plan_type,
                'monthly_data_gb': round(monthly_data_usage, 2),
                'monthly_minutes': int(monthly_minutes),
                'ott_usage_hours': round(ott_usage_hours, 1),
                'arpu': round(arpu, 2),
                'data_allowance_gb': plan_info['data_gb'],
                'minutes_allowance': plan_info['minutes']
            })
        
        return pd.DataFrame(usage_billing)
    
    def generate_crm_data(self, demographics_df: pd.DataFrame) -> pd.DataFrame:
        """Generate CRM data including satisfaction and complaints."""
        print("Generating CRM data...")
        
        crm_data = []
        
        for _, customer in demographics_df.iterrows():
            # Satisfaction score (1-10) based on tenure and demographics
            base_satisfaction = 7.5
            
            # Tenure effect (longer tenure generally = higher satisfaction)
            tenure_effect = min(1.5, customer['tenure_months'] / 24)
            
            # Age effect (middle-aged customers typically more satisfied)
            if 30 <= customer['age'] <= 50:
                age_effect = 0.5
            else:
                age_effect = -0.2
            
            satisfaction = base_satisfaction + tenure_effect + age_effect + np.random.normal(0, 1.5)
            satisfaction = max(1, min(10, satisfaction))
            
            # Complaints based on satisfaction
            complaint_prob = max(0.01, (10 - satisfaction) / 10 * 0.3)
            num_complaints = np.random.poisson(complaint_prob * 12)  # Per year
            
            # Support tickets
            support_tickets = np.random.poisson(2) + num_complaints
            
            # Payment history (based on income and satisfaction)
            payment_score = 85 + (customer['annual_income'] / 1000) * 0.1 + satisfaction * 2
            payment_score = max(300.0, min(850.0, float(payment_score + np.random.normal(0, 50))))
            
            # Late payments
            late_payments = max(0, int(np.random.exponential(1) * (10 - satisfaction) / 10))
            
            crm_data.append({
                'customer_id': customer['customer_id'],
                'satisfaction_score': round(satisfaction, 1),
                'num_complaints_12m': num_complaints,
                'support_tickets_12m': support_tickets,
                'payment_score': int(payment_score),
                'late_payments_12m': late_payments
            })
        
        return pd.DataFrame(crm_data)
    
    def generate_campaign_data(self, demographics_df: pd.DataFrame) -> pd.DataFrame:
        """Generate campaign exposure and conversion data."""
        print("Generating campaign data...")
        
        campaign_data = []
        
        # Define campaign types
        campaigns = [
            'Email_Promotion', 'SMS_Offer', 'Social_Media', 'TV_Commercial', 
            'Online_Display', 'Search_Ads', 'Direct_Mail', 'Referral_Program'
        ]
        
        for _, customer in demographics_df.iterrows():
            # Number of campaigns customer was exposed to (based on age and income)
            if customer['age'] < 35:
                base_exposure = 8  # Digital natives see more campaigns
            elif customer['age'] > 60:
                base_exposure = 4  # Less digital exposure
            else:
                base_exposure = 6
            
            num_campaigns = np.random.poisson(base_exposure)
            exposed_campaigns = np.random.choice(campaigns, size=min(num_campaigns, len(campaigns)), replace=False)
            
            total_impressions = 0
            total_clicks = 0
            total_conversions = 0
            
            for campaign in exposed_campaigns:
                # Impressions based on campaign type
                if campaign in ['Social_Media', 'Online_Display', 'Search_Ads']:
                    impressions = np.random.poisson(15)
                elif campaign in ['Email_Promotion', 'SMS_Offer']:
                    impressions = np.random.poisson(5)
                else:
                    impressions = np.random.poisson(3)
                
                # Click-through rate based on age and campaign type
                if customer['age'] < 35 and campaign in ['Social_Media', 'Online_Display']:
                    ctr = 0.05
                elif campaign in ['Email_Promotion', 'Search_Ads']:
                    ctr = 0.03
                else:
                    ctr = 0.02
                
                clicks = np.random.binomial(impressions, ctr)
                
                # Conversion rate based on income and campaign quality
                base_conversion_rate = 0.1
                if customer['annual_income'] > 70000:
                    base_conversion_rate *= 1.3
                
                conversions = np.random.binomial(clicks, base_conversion_rate)
                
                total_impressions += impressions
                total_clicks += clicks
                total_conversions += conversions
            
            campaign_data.append({
                'customer_id': customer['customer_id'],
                'campaigns_exposed': len(exposed_campaigns),
                'total_impressions': total_impressions,
                'total_clicks': total_clicks,
                'total_conversions': total_conversions,
                'primary_channel': exposed_campaigns[0] if exposed_campaigns.size > 0 else 'None'
            })
        
        return pd.DataFrame(campaign_data)
    
    def generate_digital_engagement(self, demographics_df: pd.DataFrame) -> pd.DataFrame:
        """Generate web/app engagement data."""
        print("Generating digital engagement data...")
        
        engagement_data = []
        
        for _, customer in demographics_df.iterrows():
            # Digital engagement based on age
            if customer['age'] < 30:
                base_sessions = 25
                avg_duration = 8
            elif customer['age'] < 50:
                base_sessions = 15
                avg_duration = 12
            else:
                base_sessions = 8
                avg_duration = 15
            
            # Monthly sessions
            monthly_sessions = max(0, int(np.random.poisson(base_sessions)))
            
            # Average session duration (minutes)
            session_duration = max(1, np.random.normal(avg_duration, 5))
            
            # App vs web preference
            if customer['age'] < 40:
                app_preference = 0.7  # Younger prefer mobile app
            else:
                app_preference = 0.3  # Older prefer web
            
            app_sessions = int(monthly_sessions * (app_preference + np.random.normal(0, 0.2)))
            app_sessions = max(0, min(monthly_sessions, app_sessions))
            web_sessions = monthly_sessions - app_sessions
            
            # Self-service usage
            self_service_usage = np.random.beta(2, 5) * monthly_sessions
            
            engagement_data.append({
                'customer_id': customer['customer_id'],
                'monthly_web_sessions': web_sessions,
                'monthly_app_sessions': app_sessions,
                'avg_session_duration_min': round(session_duration, 1),
                'self_service_transactions': int(self_service_usage)
            })
        
        return pd.DataFrame(engagement_data)
    
    def generate_churn_labels(self, demographics_df: pd.DataFrame, 
                            usage_df: pd.DataFrame, 
                            crm_df: pd.DataFrame) -> pd.DataFrame:
        """Generate churn labels based on customer characteristics."""
        print("Generating churn labels...")
        
        # Merge data for churn prediction
        merged_df = demographics_df.merge(usage_df, on='customer_id').merge(crm_df, on='customer_id')
        
        churn_data = []
        
        for _, customer in merged_df.iterrows():
            # Churn probability based on multiple factors
            churn_prob = 0.1  # Base churn rate
            
            # Satisfaction impact (strongest predictor)
            churn_prob += (10 - customer['satisfaction_score']) * 0.05
            
            # Tenure impact (newer customers more likely to churn)
            if customer['tenure_months'] < 6:
                churn_prob += 0.15
            elif customer['tenure_months'] < 12:
                churn_prob += 0.08
            
            # ARPU impact (very low ARPU customers may churn)
            if customer['arpu'] < 30:
                churn_prob += 0.12
            elif customer['arpu'] > 80:
                churn_prob -= 0.05  # High value customers less likely to churn
            
            # Complaints impact
            churn_prob += customer['num_complaints_12m'] * 0.1
            
            # Late payments impact
            churn_prob += customer['late_payments_12m'] * 0.05
            
            # Usage pattern impact
            usage_ratio = customer['monthly_data_gb'] / customer['data_allowance_gb']
            if usage_ratio < 0.2:  # Very low usage
                churn_prob += 0.08
            elif usage_ratio > 1.2:  # Consistently over limit
                churn_prob += 0.06
            
            # Cap probability
            churn_prob = min(0.8, max(0.01, float(churn_prob)))
            
            # Generate actual churn outcome
            churned = np.random.binomial(1, churn_prob)
            
            # If churned, set churn date
            if churned:
                days_since_join = (datetime.strptime(str(customer['join_date']), '%Y-%m-%d') - self.start_date).days
                max_churn_day = min(365 * 2.5, float(abs(days_since_join)))  # Can't churn before joining
                
                # Ensure valid range for random churn date
                min_churn_day = max(30, days_since_join + 30)  # At least 30 days after joining
                max_churn_day = max(min_churn_day + 1, int(max_churn_day))
                
                if min_churn_day < max_churn_day:
                    churn_days_from_start = np.random.randint(min_churn_day, max_churn_day)
                else:
                    churn_days_from_start = min_churn_day
                    
                churn_date = (self.start_date + timedelta(days=churn_days_from_start)).strftime('%Y-%m-%d')
            else:
                churn_date = None
            
            churn_data.append({
                'customer_id': customer['customer_id'],
                'churned': churned,
                'churn_date': churn_date,
                'churn_probability': round(churn_prob, 3)
            })
        
        return pd.DataFrame(churn_data)
    
    def generate_complete_dataset(self) -> Dict[str, pd.DataFrame]:
        """Generate complete synthetic telecom dataset."""
        print(f"Generating synthetic telecom dataset for {self.n_customers} customers...")
        
        # Generate all components
        demographics = self.generate_customer_demographics()
        usage_billing = self.generate_usage_and_billing(demographics)
        crm_data = self.generate_crm_data(demographics)
        campaign_data = self.generate_campaign_data(demographics)
        engagement_data = self.generate_digital_engagement(demographics)
        churn_data = self.generate_churn_labels(demographics, usage_billing, crm_data)
        
        # Create master dataset
        master_df = (demographics
                    .merge(usage_billing, on='customer_id')
                    .merge(crm_data, on='customer_id')
                    .merge(campaign_data, on='customer_id')
                    .merge(engagement_data, on='customer_id')
                    .merge(churn_data, on='customer_id'))
        
        print(f"Dataset generation complete! Shape: {master_df.shape}")
        
        return {
            'master_dataset': master_df,
            'demographics': demographics,
            'usage_billing': usage_billing,
            'crm_data': crm_data,
            'campaign_data': campaign_data,
            'engagement_data': engagement_data,
            'churn_data': churn_data
        }

def main():
    """Generate and save synthetic telecom dataset."""
    generator = TelecomDataGenerator(n_customers=10000)
    datasets = generator.generate_complete_dataset()
    
    # Save datasets
    import os
    data_dir = "data/raw"
    os.makedirs(data_dir, exist_ok=True)
    
    for name, df in datasets.items():
        filepath = os.path.join(data_dir, f"{name}.csv")
        df.to_csv(filepath, index=False)
        print(f"Saved {name} to {filepath}")
    
    # Print summary statistics
    master_df = datasets['master_dataset']
    print("\n=== Dataset Summary ===")
    print(f"Total customers: {len(master_df)}")
    print(f"Churn rate: {master_df['churned'].mean():.2%}")
    print(f"Average ARPU: ${master_df['arpu'].mean():.2f}")
    print(f"Average satisfaction: {master_df['satisfaction_score'].mean():.1f}/10")
    print(f"Plan distribution:")
    print(master_df['plan_type'].value_counts(normalize=True))

if __name__ == "__main__":
    main()