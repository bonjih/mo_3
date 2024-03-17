import csv
import pandas as pd
import matplotlib.pyplot as plt
import json
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA


def plot_histo(csv_file):
    df = pd.read_csv(csv_file)
    scaler = StandardScaler()
    df['movement_instant_normalized'] = scaler.fit_transform(df[['motion_sum']])

    # Plot histogram
    plt.hist(df['movement_instant_normalized'], bins=50, color='blue')
    plt.grid(True)
    plt.xlabel('Normalized Movement Instant')
    plt.ylabel('Frequency')
    plt.title('Histogram of Normalized Movement Instant')
    plt.show()


# plot_histo('output.csv')


def add_arima_trend(csv_file, order=(1, 1, 1), resample_freq='1min'):
    # Load data from CSV
    data = pd.read_csv(csv_file)

    # Set 'ts' column as the index
    data.set_index('ts', inplace=True)

    # attempt to resample data with frequency
    data_resampled = data.resample(resample_freq).mean()
    data_resampled.index = pd.date_range(start=data_resampled.index.min(), periods=len(data_resampled),
                                         freq=resample_freq)
    data_resampled.dropna(inplace=True)

    model = ARIMA(data_resampled['movement_instant'], order=order)
    model_fit = model.fit()

    print(model_fit.summary())
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    residuals.plot(kind='kde')
    pyplot.show()
    print(residuals.describe())

    trend = model_fit.predict(typ='levels')

    # original data and trend
    plt.figure(figsize=(10, 6))
    plt.plot(data_resampled.index, data_resampled['movement_instant'], label='Original Data')
    plt.plot(data_resampled.index, trend, label='Trend')
    plt.xlabel('Time')
    plt.ylabel('Movement Instant')
    plt.title('Original Data with ARIMA Trend')
    plt.legend()
    plt.show()

    # autocorrelation
    plt.figure(figsize=(10, 6))
    pd.plotting.autocorrelation_plot(data_resampled['movement_instant'])
    plt.title('Autocorrelation of Movement Instant')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.show()

    return trend


# trend = add_arima_trend('./out/datalog.csv', order=(1, 1, 1))


def write_to_csv(output_data, output_file):
    with open(output_file, mode='a', newline='') as csvfile:
        fieldnames = output_data[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerows(output_data)


def json_file(file):
    with open(file, ) as fh:
        dic = json.load(fh)

    return dic
