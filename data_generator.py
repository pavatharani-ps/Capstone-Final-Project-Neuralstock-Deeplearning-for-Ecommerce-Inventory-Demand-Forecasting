import pandas as pd
import numpy as np
import os

os.makedirs("../data/raw", exist_ok=True)

np.random.seed(42)

dates = pd.date_range(start='2023-01-01', periods=365)

categories = ['Electronics', 'Fashion', 'Groceries']

data = []

for date in dates:

    for category in categories:

        sales = np.random.randint(50, 200)

        promotion = np.random.choice([0, 1])

        data.append([date, category, sales, promotion])

df = pd.DataFrame(
    data,
    columns=['date', 'category', 'sales', 'promotion']
)

df.to_csv("../data/raw/ecommerce_inventory_demand.csv", index=False)

print("Dataset Generated Successfully")