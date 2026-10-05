import pandas as pd
import numpy as np

# Answer 1
# print(pd.__version__)

df = pd.read_csv("car_fuel_efficiency_2026.csv")

# Answer 2: the value of stop
# print(df.index)

# Answer 3
# print(df['fuel_type'].nunique())

# Answer 4
# print((df.isnull().sum().values > 0).sum())

# Answer 5
# print(df.groupby('origin').fuel_efficiency_mpg.max())

# Answer 6: corrected the answer it was wrong previously.
initial_median_horsepower = df.horsepower.median()
most_freq_horsepower = df.horsepower.mode()[0]

new_df = df
new_df.horsepower = df.horsepower.fillna(most_freq_horsepower)

final_median_horsepower = new_df.horsepower.median()

print(f"""Initial Median HP: {initial_median_horsepower}
Final Median HP: {final_median_horsepower}
""")

# Answer 7
X = df[df.origin == 'Asia'][['vehicle_weight', 'model_year']][:7].values
XTX = X.T.dot(X)
inv_XTX = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = (inv_XTX.dot(X.T)).dot(y)

# print(w.sum())

# print(df.head(n=2))