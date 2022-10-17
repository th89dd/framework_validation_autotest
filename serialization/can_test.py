# -*- coding: utf-8 -*-
"""
:version: 1.0
:copyright: 2022, Tim Häberlein, TU Dresden, FZM
"""
import time

# ---------- import block --------------
import pandas as pd
import numpy as np
# ---------- /import block -------------

# Function to calc frame_length and data_rates [see ISO 11898-1 + ISO 11898-2]
# incl. bit_stuffing [see also other papers in biblografie; Tindell et. all]
def calc_data_rate(f_bit=500, n_data_bytes=1, base_or_extended='base', print_data=False) -> dict:
    """
    Calc max./min. datarates, max./min. frame_length & max./min. frame_times
    :param f_bit: datarate in kbit/s
    :param n_data_bytes: databytes for can frame (1-8)
    :param base_or_extended: base - for classical, extended for extended
    :param print_data: True for print to stdout
    :return: result_dict
    """

    f_bit = f_bit * (10**3)
    n_trailer1 = 15
    n_trailer2 = 12
    if base_or_extended == 'base':
        n_header = 19
    else:
        n_header = 39

    # frame length
    n_frame_max = n_header + n_trailer1 + n_trailer2 + 8 * n_data_bytes + np.floor((n_header + n_trailer1 + 8 * n_data_bytes) / 5)
    n_frame_min = n_header + n_trailer1 + n_trailer2 + 8 * n_data_bytes

    # data rate
    t_frame_min = n_frame_min / f_bit
    t_frame_max = n_frame_max / f_bit

    f_data_max = 8 * n_data_bytes / t_frame_min
    f_data_min = 8 * n_data_bytes / t_frame_max
    if print_data:
        print(
            '{}_frame with {} byte max: \t n_frame: {:>#6.2f} bit'
            ' \t t_frame: {:>#6.4f} \u03BCs \t f_data: {:>#4.2f} kbit/s'.format(
                base_or_extended, n_data_bytes, n_frame_max, t_frame_max*10**6, f_data_max/10**3)
        )

        print(
            '{}_frame with {} byte min: \t n_frame: {:>#6.2f} bit'
            ' \t t_frame: {:>#6.4f} \u03BCs \t f_data: {:>#4.2f} kbit/s'.format(
                base_or_extended, n_data_bytes, n_frame_min, t_frame_min*10**6, f_data_min/10**3)
        )

    data_dict = {
        'frame_format': base_or_extended,
        'n_data_bytes': n_data_bytes,
        'n_frame_max [bit]': n_frame_max,
        't_frame_max [ms]': t_frame_max*10**3,
        'f_data_min [kbit/s]': f_data_min/10**3,
        'n_frame_min [bit]': n_frame_min,
        't_frame_min [ms]': t_frame_min*10**3,
        'f_data_max [kbit/s]': f_data_max/10**3
    }
    return data_dict

# Function to calc max wait time for can_frames
# [see Zimmermann und Schmidgall, Bussysteme in der Fahrzeugtechnik, 2014]
def calc_t_wait(t, t_frame: float, df_messages_low, df_messages_high) -> float:
    """
    calc the wait time for a can frame with given data
    :param t: cycle time / frequenz for the frame
    :param t_frame: max. frame_time from (calc_data_rate)
    :param df_messages_low: low_prior messages
    :param df_messages_high: high_prior messages
    :return: t_wait rounded with 1 digit
    """
    t_frame_low_max = df_messages_low['t_frame'].max()

    x = 0
    while x < 5*t:
        t_wait = t_frame_low_max
        for index, row in df_messages_high.iterrows():
            t_wait += ((np.ceil(x/row['t'])) * row['t_frame'])
        t_wait = round(t_wait, 1)
        if t_wait == x:
            return t_wait + t_frame
        else:
            x = round(x + 0.1, 1)
    return np.nan

# Function to calc a table with wait_times for a known_can_matrix
def calc_realtime(f_bit=500, n_bytes_list=None, t_message_list=None) -> pd.DataFrame:
    """
    Calc a table of frame_times with bus load for given lists
    :param f_bit: baud_rate in kbit/s
    :param n_bytes_list: list of number of bytes ordered by can_id
    :param t_message_list: list of cycle times for the frames
    :return: dataframe with calc values
    """
    if t_message_list is None:
        t_message_list = [10, 1, 1, 2, 2, 5, 5, 5, 5, 10, 10, 20, 20, 50, 50, 100]
    if n_bytes_list is None:
        n_bytes_list = [8, 4, 1, 8, 8, 2, 2, 7, 7, 8, 8, 6, 4, 2, 1, 4]
    message_list = list()
    for x in range(1, len(n_bytes_list)):
        n_bytes = n_bytes_list[x-1]
        t_message = t_message_list[x-1]
        # n_bytes = np.random.randint(low=1, high=8, size=1)[0]
        # t_message = [1, 5, 10, 20, 50, 100][(np.random.randint(low=0, high=4, size=1)[0])]
        message_list.append({'can_id': x, 'n_bytes': n_bytes, 't': t_message,
                             't_frame': calc_data_rate(f_bit=f_bit, n_data_bytes=n_bytes, base_or_extended='extended')['t_frame_max [ms]']})

    df_messages = pd.DataFrame(message_list)
    t_wait_list = list()
    bus_last_list = list()
    for index, row in df_messages.iterrows():
        n_bytes = row['n_bytes']
        t_frame = row['t_frame']
        df_messages_high = df_messages.iloc[:index].reset_index()
        df_messages_low = df_messages.iloc[index:].reset_index()
        t_wait_list.append(calc_t_wait(row['t'], row['t_frame'], df_messages_low, df_messages_high))
        # calc bus_last
        x = df_messages.iloc[:index+1]['t_frame'] / df_messages.iloc[:index+1]['t']
        bus_last_list.append(x.sum()*100)
    df_messages['t_wait'] = t_wait_list
    df_messages['buslast'] = bus_last_list

    return df_messages


if __name__ == '__main__':
    t0 = time.time()
    x = calc_realtime(500)
    print(time.time() - t0)
