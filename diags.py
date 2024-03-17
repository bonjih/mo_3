from matplotlib.ticker import MaxNLocator
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def my_load_and_transform(file):
    df = pd.read_csv(file)
    df["ts"] = df['ts'].to_datetime(unit='s')
    return df


def diags_window(window_size_stop=int(12.5 * 10), window_size_re=int(5 * 12.5)):
    df = pd.read_csv('./out/datalog.csv')
    df_filtered = df.sort_values(by='pos_msec')
    df_filtered['pos_sec'] = df['pos_msec'] / 1000.0
    # df['ts'] = pd.to_datetime(df['ts'], unit='s')

    df_filtered['motion_sma_rising_edge'] = df_filtered['movement_instant'].rolling(window=window_size_re).mean()
    df_filtered['motion_sma_stable'] = df_filtered['movement_instant'].rolling(window=window_size_stop).mean()
    df_filtered['motion_sma_var_rising_edge'] = df_filtered['movement_instant'].rolling(window=window_size_re).var()
    df_filtered['motion_sma_var_stop'] = df_filtered['movement_instant'].rolling(window=window_size_stop).var()
    df_filtered['inst_diff1'] = df_filtered['movement_instant'].diff(13).rolling(window=window_size_re).mean()

    scaler = MinMaxScaler()
    plt.figure(figsize=(40, 15))

    plt.scatter(x=df_filtered['pos_sec'], y=df_filtered[['movement_instant']], s=8, marker='+', label='Motion Sum')
    # plt.plot(df_filtered['pos_sec'], scaler.fit_transform(df_filtered[['motion_sma_rising_edge']]), label=f'SMA (Window Size = {window_size_re})', color='red')
    # plt.plot(df_filtered['pos_sec'], scaler.fit_transform(df_filtered[['motion_sma_stable']]), label=f'SMA (Window Size = {window_size_stop})', color='orange')
    # plt.plot(df_filtered['pos_sec'], scaler.fit_transform(df_filtered[['motion_sma_var_rising_edge']]), label=f'Var (Window Size = {window_size_re})', color='green')
    plt.plot(df_filtered['pos_sec'], df_filtered[['auto1sec']], label=f'absolute diff of sum between n-13th frame',
             color='purple')

    plt.axhline(y=100000, color='magenta', linestyle='--', linewidth=1)

    plt.xlabel('Timestamp')
    plt.ylabel('Motion Sum')
    plt.legend()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    df_filtered.to_csv("./out/df_filtered.csv")


def diags_sma():
    df = pd.read_csv('./out/output.csv')
    plt.figure(figsize=(40, 15))

    df_scaled = df.copy()
    df_scaled['scaled_motion'] = (df_scaled['motion_sum'] - df_scaled['motion_sum'].min()) / (
            df_scaled['motion_sum'].max() - df_scaled['motion_sum'].min())
    df_scaled['scaled_sma'] = (df_scaled['motion_sma'] - df_scaled['motion_sum'].min()) / (
            df_scaled['motion_sum'].max() - df_scaled['motion_sum'].min())

    plt.scatter(x=df_scaled.index, y=df_scaled['scaled_motion'], s=8, marker='+', label='Scaled Motion Sum')
    plt.plot(df_scaled.index, df_scaled['scaled_sma'], label='Scaled Motion SMA', color='red')

    plt.xlabel('Index')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.savefig("threshold_plot3.png")
    plt.show()


diags_window()
