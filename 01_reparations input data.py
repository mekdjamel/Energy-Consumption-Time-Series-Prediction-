import pickle
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime


def parser(x):
    return datetime.strptime('' + x, '%Y-%m-%d')
opsd_daily = pd.read_csv('opsd_germany_daily.csv',parse_dates=[0], date_parser=parser)
print(opsd_daily.head(3))
print(opsd_daily.shape) # shape of (4383, 5)

# Prepare the consumption data
opsd_daily_Consumption = pd.DataFrame(data=opsd_daily['Consumption'], index=opsd_daily.index)
print(opsd_daily_Consumption.head(3))
print(opsd_daily_Consumption.shape) # shape of (4383, 1)

# Splitting Data for Training and Testing
train_size = int(len(opsd_daily_Consumption) * 0.8)
test_size = len(opsd_daily_Consumption) - train_size
train_opsd_daily_Consumption, test_opsd_daily_Consumption = opsd_daily_Consumption[0:train_size], opsd_daily_Consumption[train_size:len(opsd_daily_Consumption)]
print("train_opsd_daily_Consumption.shape: ", train_opsd_daily_Consumption.shape)
print("test_opsd_daily_Consumption.shape: ", test_opsd_daily_Consumption.shape)

_ = test_opsd_daily_Consumption.rename(columns={'Consumption': 'TEST SET'}) \
    .join(train_opsd_daily_Consumption.rename(columns={'Consumption': 'TRAINING SET'}), how='outer') \
    .plot(figsize=(15,5), title='Daily Energy Consumption Time Series Data')
plt.grid()
plt.show()

# save the data into pickle
with open('Train-Test_opsd_daily_Consumption.pkl', 'wb') as f:
    pickle.dump([train_opsd_daily_Consumption, test_opsd_daily_Consumption], f)
