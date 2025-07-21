# generate_sample_data.py
import pandas as pd
import numpy as np

np.random.seed(42)

N = 500

data = {
    'monthly_topups': np.random.randint(1, 30, N),
    'sms_sent': np.random.randint(0, 100, N),
    'on_time_payments': np.random.randint(0, 12, N),
    'total_payments': 12,
    'ecommerce_transactions': np.random.randint(0, 50, N),
    'social_posts': np.random.randint(0, 75, N),
    'label': np.random.choice([0, 1], N, p=[0.35, 0.65])
}

df = pd.DataFrame(data)
df['payment_regular'] = df['on_time_payments'] / df['total_payments']
df['digital_activity_score'] = (df['ecommerce_transactions'] + df['social_posts']) / 2
df.to_csv('data/sample_credit_data.csv', index=False)
