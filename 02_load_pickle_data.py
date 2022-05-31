import pickle
import matplotlib.pyplot as plt

# load data
train_opsd_daily_Consumption, test_opsd_daily_Consumption = pickle.load(open('Train-Test_opsd_daily_Consumption.pkl', "rb"))

_ = test_opsd_daily_Consumption.rename(columns={'Consumption': 'TEST SET'}) \
    .join(train_opsd_daily_Consumption.rename(columns={'Consumption': 'TRAINING SET'}), how='outer') \
    .plot(figsize=(15,5), title='Overall Charge')
plt.grid()
plt.show()
