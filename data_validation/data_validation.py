import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import correlate
from scipy.signal import correlation_lags
from datetime import datetime as dt


def corr_matrix(x, y, max_index=90):
    """
    Function to calculate correlation

    :param x:
    :param y:
    :param max_index:
    :return:
    """
    try:
        shortest = min(x.shape[0], y.shape[0])
    except Exception:
        shortest = max_index
    return np.corrcoef(x.iloc[:shortest].values, y.iloc[:shortest].values)


def plot_correlation(x, y, text):
    """
    Function to plot time series and show the correlation

    :param x:
    :param y:
    :param text:
    :return:
    """
    # plot
    fig, ax = plt.subplots(figsize=(12, 6))
    # x.plot(label="x", ax=ax)
    x.plot(ax=ax)
    # y.plot(label="y", ax=ax)
    y.plot(ax=ax)
    plt.title(f"Correlation {text}: {corr_matrix(x, y)[0, 1]}")
    ax.legend(loc="best")
    ax.grid()
    plt.show()


def shift_dataframe(df: pd.DataFrame, column_name: str, time_shift: float):
    """
    shift a given column of a dataframe by given time_shift

    :param df: the dataframe
    :param column_name: the name of the column
    :param time_shift: the time_shift value
    :return: interpolated dataframe with additionally column of shifted values
    """

    shifted = df[[column_name]].copy().dropna().shift(
        periods=round(time_shift,9), freq="S").rename(columns={column_name: f'{column_name}_shifted'})
    # df[f'{column_name}_shifted'] = shifted
    return pd.concat([df, shifted], axis=1).interpolate(method='linear')


def calc_time_shift_for_dataframe(df: pd.DataFrame) -> float:
    """
    calculate the time_shift between first 2 columns of the dataframe

    :param df: dataframe
    :return: time_shift
    """
    # do cross correlation and calc lag
    x = df.iloc[0:-1, 0].dropna().reset_index(drop=True)
    y = df.iloc[0:-1, 1].dropna().reset_index(drop=True)

    correlation = correlate(x, y, mode='full')
    lags = correlation_lags(x.size, y.size, mode="full")
    lag = lags[np.argmax(correlation)]

    # calc time_diff
    time_diff = df.index.to_series().diff().dt.total_seconds().div(1, fill_value=0)
    time_shift = time_diff.mean()*lag

    print(f'time shift is: \t {time_shift}')

    return time_shift


def shift_for_maximum_correlation(x, y) -> tuple:
    """
    Function to calculate cross-correlation, extract the best matching shift
    and then shift one of the series appropriately.

    :param x: Series 1
    :param y: Series 2
    :return: shifted series 1 & 2 and the lag as tuple
    """

    correlation = correlate(x, y, mode="full")
    lags = correlation_lags(x.size, y.size, mode="full")
    lag = lags[np.argmax(correlation)]
    print(f"Best lag: {lag}")
    if lag < 0:
        y = y.iloc[abs(lag):].reset_index(drop=True)
    else:
        x = x.iloc[lag:].reset_index(drop=True)
    return x, y, lag


def plot_dataframe(df: pd.DataFrame, x_label='datetime', figsize=(12,6), rotation=45,
                   y_label='vehicle_speed [km/h]', save=False, name='test'):
    """
    plot function for a dataframe

    :param df: pandas dataframe to plot
    :param x_label: label of x-axis
    :param y_label: label of y-axis
    :param save: bool, true if plot should be saved as file
    :param name: name of the saved file
    :return: -
    """
    fig, ax = plt.subplots(figsize=figsize)
    # df.plot(ax=ax, marker='.', ms='6')
    df.plot(ax=ax, legend=False, kind='line', lw=1.2, ls='-', marker='', markersize=2)
    # df.plot()
    # plt.plot(df)
    # Show the major grid lines with dark grey lines
    ax.set_xlabel(x_label, fontsize='large', fontweight='medium', labelpad=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_ylabel(y_label, fontsize='large', fontweight='medium', labelpad=10)

    ax.grid(visible=True, which='major', color='#666666', linestyle='-')
    # Show the minor grid lines with very faint and almost transparent grey lines
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    ax.legend(loc='best', ncol=6, fancybox=True, shadow=True, fontsize='large')  # bbox_to_anchor=(0.5, 1.05),

    # set x_ticks
    plt.xticks(fontsize='medium', rotation=rotation)  # rotation=45)



    plt.tight_layout()
    #fig.autofmt_xdate()
    plt.show()

    if save:
        fig.savefig('data_validation/_output/{name}_{date}.svg'.format(name=name, date=dt.now().strftime('%Y_%m_%d')),
                    format='svg', dpi=2400, transparent=False)


def plot_dataframes(dfs: list, x_label: str = None, y_labels: list = None, max_y: float = None,
                   lw: float = 1.2, m_size: float = 4, fig_size: tuple = (15, 8), log_x: bool = False,
                   save: bool = False, name: str = 'test', rolling: int = None):
    """
    plot function for a dataframe

    :param rolling: calc rolling mean of given value (int)
    :param m_size: marker size
    :param max_y: cut of y
    :param lw: line width
    :param fig_size: size of the plot
    :param df: list of pandas dataframe to plot
    :param x_label: label of x-axis
    :param y_label: label of y-axis
    :param save: bool, true if plot should be saved as file
    :param name: name of the saved file
    :return: -
    """

    n_plots = len(dfs)

    fig, axs = plt.subplots(nrows=n_plots, ncols=1, sharex='all', figsize=fig_size)
    if n_plots == 1:
        axs = [axs]
    for df, ax, y_label in zip(dfs, axs, y_labels):
        if rolling is not None:
            df = df.rolling(rolling).mean()
        df.plot(ax=ax, legend=False, kind='line', lw=lw, ls='-', marker='.', markersize=m_size, logx=log_x)
        ax.set_ylabel(y_label, fontsize='large', fontweight='medium', labelpad=10)
        if max_y is not None:
            ax.set_ylim(top=max_y)

        ax.set_xlabel(x_label, fontsize='large', fontweight='medium', labelpad=10)
        # Show the major grid lines with dark grey lines
        # Show the minor grid lines with very faint and almost transparent grey lines
        ax.grid(visible=True, which='major', color='#666666', linestyle='-')
        ax.minorticks_on()
        ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    axs[0].legend(loc='best', ncol=6, fancybox=True, shadow=True, fontsize='large')  # bbox_to_anchor=(0.5, 1.05),

    # set x_ticks
    plt.xticks(fontsize='medium')  # rotation=45)
    plt.tight_layout()
    plt.show()
    if save:
        fig.savefig('data_validation/_output/{name}_{date}.png'.format(name=name, date=dt.now().strftime('%Y_%m_%d')),
                    format='png', dpi=2400, transparent=False)


if __name__ == '__main__':
    # read data
    # and print info from dataframe
    df_vgl = pd.read_parquet('speed_comparison_2019_11_06.parquet.gzip')
    print(df_vgl.info())

    # extract values for interpolation
    v_db_origin = df_vgl[['v_db', 'timestamp']].dropna()
    v_blf_origin = df_vgl[['v_blf', 'timestamp']].dropna()

    # interpolate data & plot
    df = v_blf_origin
    df['v_db_interp'] = np.interp(v_blf_origin.timestamp, v_db_origin.timestamp, v_db_origin.v_db)
    print(df.info())
    plot_dataframe(df[['v_blf', 'v_db_interp']], save=False)
    plot_dataframe(df[['v_blf', 'v_db_interp']].between_time('11:25', '11:30'))
    time_shift = calc_time_shift_for_dataframe(df[['v_blf', 'v_db_interp']])
    df_shifted = shift_dataframe(df, 'v_db_interp', time_shift)
    plot_dataframe(df_shifted[['v_blf', 'v_db_interp_shifted']].between_time('11:25', '11:27'))

    # plot for paper
    df = df.rename(columns={'v_blf': 'CANoe', 'v_db_interp': 'framework'})
    plot_dataframe(df[['CANoe', 'framework']], name='speed_comparison_interpolated', save=True, figsize=(9, 3))
    time_zoom = '11:25:00', '11:25:45'
    df_shifted = df_shifted.rename(columns={'v_blf': 'CANoe', 'v_db_interp': 'framework', 'v_db_interp_shifted': 'framework_shifted'})
    plot_dataframe(df_shifted[['CANoe', 'framework', 'framework_shifted']].between_time(*time_zoom), figsize=(9, 3), save=True, name='speed_comparison_correlation_zoom')

