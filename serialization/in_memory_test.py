# -*- coding: utf-8 -*-
"""
:version: 1.0
:copyright: 2022, Tim Häberlein, TU Dresden, FZM
"""
# ---------- import block --------------
import sys
import psutil, os
import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime as dt

from serialization.CANBase import CocaplCanFrame
import time
from superjson import json as sjson
import gzip
import asn1

from protobuf3.message import Message
from protobuf3.fields import StringField, Int32Field, MessageField, FloatField
# ---------- /import block --------------


class ProtoCanFrame(Message):
    id = StringField(field_number=1, required=False)
    length = Int32Field(field_number=2, required=False)
    type = StringField(field_number=3, required=False)
    data = StringField(field_number=4, required=False)
    ts = FloatField(field_number=5, required=False)
    vin = StringField(field_number=6, required=False)
    ch = StringField(field_number=7, required=False)


class ProtoCanFrames(Message):
    can_frames = MessageField(field_number=1, repeated=True, message_cls=ProtoCanFrame)


class Serializer:
    """
    A class that represents a serializer
    """

    def __init__(self, frame_number: int = 10,
                 external_data: list = None, automode=True):
        """
        Init a serializer object

        :param frame_number: number of CAN_frames for test data
        :param external_data: external_data to serialize
        """

        self.__frame_number = frame_number
        self.__t_gen_data = None
        self.__t_serialize = None
        self.__t_deserialize = None
        self.__serialized_data = None
        self.__deserialized_data = None

        # make data if there is no external
        if external_data is None:
            self.__frame_list = self.gen_data()
        else:
            self.__frame_list = external_data

        self.__name = self._set_name()

        if automode:
            self.do_serialize()
            self.do_deserialization()

    def gen_data(self) -> list:
        """
        method to generate test data as a list
        :return: frame_list
        """
        t0 = time.time()
        data_list = list()
        for i in range(0, self.__frame_number):
            data_list.append(CocaplCanFrame().raw_data)  # append a clear dict for frame_data
        self.__t_gen_data = time.time() - t0

        return data_list

    def do_serialize(self):
        """
        serialize the data
        :return: -
        """
        t0 = time.time()
        self.__serialized_data = self._serialize(self.__frame_list)
        self.__t_serialize = time.time() - t0

    def do_deserialization(self):
        """
        deserialize the data
        :return:
        """
        t0 = time.time()
        self.__deserialized_data = self._deserialize(self.__serialized_data)
        self.__t_deserialize = time.time() - t0

    def print(self):
        """
        print to stdout time and size for serialization
        :return: -
        """
        print('{name:<15} \t raw_size: {raw_size:>#14.6f} kB \t '
              'serialized: {ser_size:>#18.6f} kB in {t_ser:>#12.8f} sec \t '
              'deserialized: {dser_size:>#14.6f} kB in {t_deser:>#12.8f} sec'.format(

            name=self.__name, raw_size=self.gen_data_size,
            ser_size=self.serialized_data_size, t_ser=self.t_serialize,
            dser_size=self.deserialized_data_size, t_deser=self.t_deserialize))

    def print_raw(self):
        """
        print to stdout time and size for raw_data
        :return: -
        """
        print('{name:<15} \t {data_size:<14} kB  in {t:<12} sec'.format(
            name='raw_data', data_size=self.gen_data_size, t=self.t_gen_data
        ))

    def check_data(self):
        """
        method to check frame_list and deserialized data
        :return: True if data is == else False
        """
        if not self.__deserialized_data:
            self.do_deserialization()
        return self.__frame_list == self.__deserialized_data

    # -----------------------------------------------------------------------------
    # private methods to overwrite with sub-classes
    # -----------------------------------------------------------------------------
    @staticmethod
    def _serialize(data):
        """
        internal method to do the specific serialization
        :param data:
        :return: serialized data
        """
        pass

    @staticmethod
    def _deserialize(data):
        """
        internal method to do the specific deserialization

        :param data:
        :return: deserialized data
        """
        pass

    @staticmethod
    def _set_name():
        return 'super_class'

    # -----------------------------------------------------------------------------
    # properties
    # -----------------------------------------------------------------------------
    @property
    def frame_number(self) -> int:
        """
        Return the number of frames for test data
        :return: frame number
        """
        return self.__frame_number

    @property
    def frame_list(self) -> list:
        """
        Returns the data
        :return: frame_list
        """
        return self.__frame_list

    @property
    def serialized_data(self) -> bytes:
        """
        Returns the serialized data
        :return: serialized_data
        """
        return self.__serialized_data

    @property
    def deserialized_data(self) -> list:
        """
        Returns the deserialized data
        :return: deserialized data
        """
        return self.__deserialized_data

    @property
    def t_serialize(self) -> float:
        """
        Returns the time to serialize data
        :return: t_serialize_data in sec
        """
        return self.__t_serialize

    @property
    def t_deserialize(self) -> float:
        """
        Returns the time to deserialize data
        :return: time in sec
        """
        return self.__t_deserialize

    @property
    def serialized_data_size(self) -> float:
        """
        Returns the data size of serialized data in kB
        :return: serialzed_data_size
        """
        return sys.getsizeof(self.__serialized_data) / 1024

    @property
    def deserialized_data_size(self) -> float:
        """
        Returns the data size of deserialized data in kB
        :return: deserialized_data_size in kB
        """
        return sys.getsizeof(self.__deserialized_data) / 1024

    @property
    def t_gen_data(self) -> float:
        """
        Returns the time to generate data
        :return: t_gen_data in sec
        """
        return self.__t_gen_data

    @t_gen_data.setter
    def t_gen_data(self, value):
        """
        Set the time to generate data
        :param value: time in sec
        :return: -
        """
        assert type(value) == float, 'type of value must be float'
        self.__t_gen_data = value

    @property
    def gen_data_size(self) -> float:
        """
        Returns the size of generated data in kB
        :return: gen_data_size
        """
        return sys.getsizeof(self.__frame_list) / 1024

    @property
    def results(self) -> tuple:
        """
        Returns the results of the serialization
        :return: t, data_size
        """
        return self.t_serialize, self.serialized_data_size

    @property
    def result_dict(self) -> dict:
        return {
            'method': self.__name,
            'raw_data_size [kB]': self.gen_data_size,
            'time_2_gen_data [ms]': self.t_gen_data * 1000,
            'serialized_data_size [kB]': self.serialized_data_size,
            'time_2_serialize [ms]': self.t_serialize * 1000,
            'deserialized_data_size [kB]': self.deserialized_data_size,
            'time_2_deserialize [ms]': self.t_deserialize * 1000,
            'number_of_frames': self.frame_number
        }

    def __repr__(self):
        return 'serializer("{name}")'.format(
            name=self.__name,
        )


class RawJson(Serializer):
    # def __init__(self, frame_number: int = 10,
    #              external_data: list = None):
    #     super().__init__(frame_number, external_data)
    #     self.name = 'raw_json'
    @staticmethod
    def _set_name():
        return 'raw_json'

    @staticmethod
    def _serialize(data):
        return json.dumps(data).encode('utf-8')

    @staticmethod
    def _deserialize(data):
        return json.loads(data)


class RawPickle(Serializer):
    @staticmethod
    def _set_name():
        return 'raw_pickle'

    @staticmethod
    def _serialize(data):
        return pickle.dumps(data)

    @staticmethod
    def _deserialize(data):
        return pickle.loads(data)


class RawSJson(Serializer):
    @staticmethod
    def _set_name():
        return "raw_sjson"

    @staticmethod
    def _serialize(data):
        return sjson.dumps(data, compress=False).encode('utf-8')

    @staticmethod
    def _deserialize(data):
        return sjson.loads(data.decode('utf-8'), decompress=False)


class RawProtob(Serializer):
    @staticmethod
    def _set_name():
        return 'raw_protob'

    def gen_data(self) -> list:
        t0 = time.time()
        data_list = list()
        for i in range(0, self.frame_number):
            proto_frame = ProtoCanFrame()
            can_frame = CocaplCanFrame().raw_data

            proto_frame.ch = can_frame['ch']
            proto_frame.id = can_frame['frame_id']
            proto_frame.length = can_frame['length']
            proto_frame.data = can_frame['data']
            proto_frame.ts = can_frame['ts']
            proto_frame.vin = can_frame['vin']
            proto_frame.type = can_frame['frame_type']

            data_list.append(proto_frame)
        self.t_gen_data = time.time() - t0

        return data_list

    @staticmethod
    def _serialize(data):
        frames = ProtoCanFrames()
        frames.can_frames.extend(data)
        return frames.encode_to_bytes()

    @staticmethod
    def _deserialize(data):
        frames = ProtoCanFrames.create_from_bytes(data)
        return [frame for frame in frames.can_frames]


class Asn1Json(Serializer):
    @staticmethod
    def _set_name():
        return 'asn1_json'

    @staticmethod
    def _serialize(data):
        encoder = asn1.Encoder()
        encoder.start()
        encoder.write(json.dumps(data))
        return encoder.output()

    @staticmethod
    def _deserialize(data):
        decoder = asn1.Decoder()
        decoder.start(data)
        return json.loads(decoder.read()[1])


class Asn1Pkl(Serializer):
    @staticmethod
    def _set_name():
        return 'asn1_pkl'

    @staticmethod
    def _serialize(data):
        encoder = asn1.Encoder()
        encoder.start()
        encoder.write(pickle.dumps(data))
        return encoder.output()

    @staticmethod
    def _deserialize(data):
        decoder = asn1.Decoder()
        decoder.start(data)
        return pickle.loads(decoder.read()[1])


class CompressedSJson(Serializer):
    @staticmethod
    def _set_name():
        return 'zip_sjson'

    @staticmethod
    def _serialize(data):
        return sjson.dumps(data, compress=True).encode('utf-8')

    @staticmethod
    def _deserialize(data):
        return sjson.loads(data.decode('utf-8'), decompress=True)


class GzipJson(Serializer):
    @staticmethod
    def _set_name():
        return 'gzip_json'

    @staticmethod
    def _serialize(data):
        return gzip.compress(json.dumps(data).encode('utf-8'), 9)

    @staticmethod
    def _deserialize(data):
        return json.loads(gzip.decompress(data).decode('utf-8'))


class GzipPkl(Serializer):
    @staticmethod
    def _set_name():
        return 'gzip_pkl'

    @staticmethod
    def _serialize(data):
        return gzip.compress(pickle.dumps(data), 9)

    @staticmethod
    def _deserialize(data):
        return pickle.loads(gzip.decompress(data))


class GzipProtob(RawProtob):
    @staticmethod
    def _set_name():
        return 'gzip_protob'

    @staticmethod
    def _serialize(data):
        frames = ProtoCanFrames()
        frames.can_frames.extend(data)
        return gzip.compress(frames.encode_to_bytes(), 9)

    @staticmethod
    def _deserialize(data):
        frames = ProtoCanFrames.create_from_bytes(gzip.decompress(data))
        return [frame for frame in frames.can_frames]


def calc_serialization_times(test_time: float = 1, frames_per_second: int = 4090):
    """
    Function to calc for all implemented methods the times and data_size
    :param test_time: measurement_time in seconds
    :param frames_per_second: can data frames sent per second
    :return: pandas dataframe with results
    """

    # calc number_of_frames
    number_of_frames = int(np.ceil(test_time * frames_per_second))
    # do serialization for all implemented methods
    method_list = [
        RawPickle(number_of_frames),
        RawJson(number_of_frames),
        RawSJson(number_of_frames),
        RawProtob(number_of_frames),
        CompressedSJson(number_of_frames),
        GzipJson(number_of_frames),
        GzipPkl(number_of_frames),
        GzipProtob(number_of_frames),
        Asn1Json(number_of_frames),
        Asn1Pkl(number_of_frames)
    ]

    result_list = list()
    time_list = list()
    for method in method_list:
        result_list.append(method.result_dict)
        time_list.append(test_time)
    df_result = pd.DataFrame(result_list)
    df_result['test_time [s]'] = time_list
    return df_result.set_index(['test_time [s]', 'method'])


def plot_serialization_data_as_bar(
        df: pd.DataFrame, x_label: str = None, y_labels: list = None,
        save: bool = False, name: str = 'test', fig_size: tuple = None):
    """
    plot function for a dataframe
    :param fig_size: size of the plot
    :param df: pandas dataframe to plot
    :param x_label: label of x-axis
    :param y_labels: list with label of y-axis
    :param save: bool, true if plot should be saved as file
    :param name: name of the saved file
    :return: -
    """
    if x_label is None:
        x_label = 'record_time [s], method'
    if y_labels is None:
        y_labels = ['time [ms]', 'data_size [kB]']
    if fig_size is None:
        fig_size = (15, 8)

    # split dataframe & rename ticks
    df_time = df[['time_2_gen_data [ms]', 'time_2_serialize [ms]', 'time_2_deserialize [ms]']]
    df_time = df_time.rename(columns={
        'time_2_gen_data [ms]': 'gen_data',
        'time_2_serialize [ms]': 'serialize',
        'time_2_deserialize [ms]': 'deserialize'
    }).sort_index()

    df_data = df[['raw_data_size [kB]', 'serialized_data_size [kB]']]
    df_data = df_data.rename(columns={
        'raw_data_size [kB]': 'raw',
        'serialized_data_size [kB]': 'serialized',
    }).sort_index()

    fontsize = 'x-large'

    # size: xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None
    # line: # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    # make fig + axis(ax):
    fig, ax_list = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=fig_size)
    # df.plot(ax=ax, lw=1.5, ls='-')

    df_time.plot(ax=ax_list[0], kind='bar', width=0.9, edgecolor="white", linewidth=0.3)  # width=0.4 df.plot.barh(ax=ax1)
    # ax1.set_xlabel(x_label, fontsize='medium', fontweight='bold', labelpad=10)
    ax_list[0].set_ylabel(y_labels[0], fontsize=fontsize, fontweight='medium', labelpad=10)

    df_data.plot(ax=ax_list[1], kind='bar', width=0.6, edgecolor="white", linewidth=0.9)
    ax_list[1].set_xlabel(x_label, fontsize=fontsize, fontweight='medium', labelpad=10)
    ax_list[1].set_ylabel(y_labels[1], fontsize=fontsize, fontweight='medium', labelpad=10)

    ax_list[0].legend(loc='best', ncol=6, fancybox=True, shadow=True, fontsize=fontsize)
    ax_list[1].legend(loc='best', ncol=6, fancybox=True, shadow=True, fontsize=fontsize)

    # Show the major grid lines with dark grey lines
    # Show the minor grid lines with very faint and almost transparent grey lines
    for ax in ax_list:
        ax.grid(visible=True, which='major', color='#666666', linestyle='-')
        ax.minorticks_on()
        ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

    # set x_ticks
    plt.xticks(fontsize=fontsize, rotation=90)  # rotation=45)
    # make some magic
    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig('serialization/_output/{name}_{date}.svg'.format(name=name, date=dt.now().strftime('%Y_%m_%d')),
                    format='svg', dpi=1200, transparent=False)


def split_df_for_single_plot(
        df: pd.DataFrame, ignore_list: list = None, rolling: int = None) -> list:
    """
    split the dataframe into 2 dataframes for plotting data_size and time separately
    :param rolling: calc a rolling mean for the given number
    :param ignore_list: list of methods to drop
    :param df: Dataframe from calc serialization
    :return: separated dataframes: (df_time, df_data_size)
    """
    # df.drop(['raw_pickle', 'raw_json'], axis=1, level=1)
    # first calc time
    df = df.reset_index().pivot(index='test_time [s]', columns='method')
    # if ignore_list is not None:
    #     df = df.drop(ignore_list, axis=1, level=1)
    df.index = np.round(df.index, 3)
    time_2_gen_data = df['time_2_gen_data [ms]'].loc[:, ['raw_json', 'raw_protob']].rename(
        columns={
            'raw_json': '[raw]',
            'raw_protob': '[raw_protob]'
        })
    # calc size
    raw_size = df['raw_data_size [kB]'].loc[:, ['raw_json', 'raw_protob']].rename(
        columns={
            'raw_json': '[raw]',
            'raw_protob': '[raw_protob]'
        })
    # calc dataframes
    df_list = list()
    df_list.append(df['time_2_serialize [ms]'].join(time_2_gen_data))  # times2serialize
    df_list.append(df['time_2_deserialize [ms]'].join(time_2_gen_data))  # times2deserialize
    df_list.append(df['serialized_data_size [kB]'].join(raw_size))  # data_size

    # add number of frames
    # and clear df by ignore_list
    frames = df['number_of_frames']['raw_json'].rename('#frames')
    dfs = list()
    for df in df_list:
        if ignore_list is not None:
            df = df.drop(ignore_list, axis=1)
        df = df.reset_index().set_index(['test_time [s]', frames])
        df = df.sort_index()
        if rolling is not None:
            df = df.rolling(rolling).mean()
        dfs.append(df)

    return dfs


def plot_serialization_data_as_line(
        df: pd.DataFrame, x_label: str = None, y_labels: list = None, logx: bool = True, logy: bool = True,
        lw: float = 1.2, m_size: float = 4, max_y_values: list = None, fig_size: tuple = None,
        save: bool = False, name: str = 'test', ignore_list: list = None, rolling: int = None):
    """
    plot function for a dataframe
    :param rolling: calc rolling mean of given value (int)
    :param m_size: marker size
    :param max_y_values: cut of y
    :param lw: line width
    :param logy:log or not for y
    :param logx: log or not for x
    :param ignore_list: list of methods to ignore (drop from dataframe)
    :param fig_size: size of the plot
    :param df: pandas dataframe to plot
    :param x_label: label of x-axis
    :param y_labels: list with label of y-axis
    :param save: bool, true if plot should be saved as file
    :param name: name of the saved file
    :return: -
    """
    if x_label is None:
        x_label = 'record_time [s], # frames'
    if y_labels is None:
        y_labels = ['time_2_serialize [ms]', 'time_2_deserialize [ms]', 'data_size [kB]']
    if fig_size is None:
        fig_size = (15, 8)
    # split dataframe & rename ticks
    if max_y_values is None:
        max_y_values = [None, None, None]
    dfs = split_df_for_single_plot(df, ignore_list=ignore_list, rolling=rolling)
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex='all', figsize=fig_size)

    for df, ax, y_label, max_y_value in zip(dfs, axs, y_labels, max_y_values):
        df.plot(
            ax=ax, legend=False, kind='line', lw=lw, ls='-', logx=logx, logy=logy,
            table=False, marker='.', markersize=m_size)
        ax.set_ylabel(y_label, fontsize='medium', fontweight='medium', labelpad=10)
        # Show the major grid lines with dark grey lines
        # Show the minor grid lines with very faint and almost transparent grey lines
        ax.grid(visible=True, which='major', color='#666666', linestyle='-')
        ax.minorticks_on()
        ax.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)

        if max_y_value is not None:
            ax.set_ylim(top=max_y_value)

    axs[0].legend(loc='best', ncol=6, fancybox=True, shadow=True)  # bbox_to_anchor=(0.5, 1.05),
    axs[2].set_xlabel(x_label, fontsize='medium', fontweight='medium', labelpad=10)

    # set x_ticks
    plt.xticks(fontsize='small')  # rotation=45)
    # make some magic
    plt.tight_layout()
    plt.show()

    if save:
        fig.savefig(
            'serialization/_output/{name}_{date}.svg'.format(
                name=name, date=dt.now().strftime('%Y_%m_%d')
            ), format='svg', dpi=1200, transparent=False)


def plot_dataframe(dfs: list, x_label: str = None, y_labels: list = None, max_y: float = None,
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
        fig.savefig('serialization/_output/{name}_{date}.svg'.format(name=name, date=dt.now().strftime('%Y_%m_%d')),
                    format='svg', dpi=1200, transparent=False)


def save_data(df: pd.DataFrame, path: str = None, name: str = 'in_memory_result'):
    """
    save dataframe
    :param name: name of the file (optional)
    :param path: path (optional)
    :param df: dataframe
    :return: -
    """
    date = dt.now().strftime('%Y_%m_%d')
    if path is not None:
        file_name = '{path}/{name}_{date}.parquet.gzip'.format(path=path, name=name, date=date)
    else:
        file_name = '{name}_{date}.parquet.gzip'.format(name=name, date=date)
    df.to_parquet(file_name, compression='gzip')


def read_data(file_name: str) -> pd.DataFrame:
    """
    read a parquet_file
    :param file_name: name of file inkl. path
    :param df: dataframe with result data
    :return: dataframe
    """

    return pd.read_parquet(file_name)


def do_test(end_time=1.0, step=0.1):
    """
    calc multiple times serialization_times
    :param end_time: max time
    :param step: step-size
    :return: result dataframe
    """
    result_list = list()
    for test_time in np.arange(0, end_time, step):
        result_list.append(calc_serialization_times(test_time))

    return pd.concat(result_list)


def do_test2(end_time: float = 1.0, step: float = 0.1, cycles: int = 2):
    """
    calc multiple times serialization_times

    :param cycles: number of cycles (calc mean value for each cycle)
    :param end_time: max time
    :param step: step-size
    :return: result dataframe
    """
    data_frame_list = list()
    for i in range(0, cycles):
        result_list = list()
        for test_time in np.arange(0, end_time, step):
            result_list.append(calc_serialization_times(test_time))
        data_frame_list.append(pd.concat(result_list))

    return pd.concat([each.stack() for each in data_frame_list], axis=1).apply(lambda x: x.mean(), axis=1).unstack()


if __name__ == '__main__':

    # set high priority
    p = psutil.Process(os.getpid())
    print('pid: ', os.getpid(), 'priority: ', p.nice())
    p.nice(psutil.REALTIME_PRIORITY_CLASS)
    print('pid: ', os.getpid(), 'priority: ', p.nice())
    # generate data and save it
    # result = do_test(60, 0.1)
    #result = do_test2(1.1, 0.01, 100)
    # save_data(result, name='in_memory_result_0to1_in0.01_100times')
    # result = do_test2(11, 0.1, 20)
    # save_data(result, name='in_memory_result_0to11_in0.1')

    result = do_test2(2, 0.5, 5)
    plot_serialization_data_as_bar(result.loc[1:2,:].sort_index(), save=True, name='serialize_comp_bar', fig_size=(15, 5))
    # save_data(result, name='in_memory_result_0to61_in0.1')

    # result = read_data('in_memory_result_0to1in0.005_5times_2022_09_14.parquet.gzip')

    # plot_serialization_data(result, fig_size=(16, 8))
    # plot_serialization_data(result.drop(
    #     index=['raw_protob', 'gzip_protob', 'raw_json', 'raw_sjson', 'asn1_json', 'compressed_sjson'],
    #     level=1), fig_size=(16, 8))

    # plot_dataframe(result_time.drop(index=['raw_protob', 'gzip_protob'], level=1))

    # result_time[['time_2_serialize [ms]']].groupby(level='test_time [s]', axis=0)
    # x = result.xs('raw_pickle', level=1, drop_level=True)
    # x = result.loc[:, ['raw_pickle', 'raw_json'], :]
    ignore = [
        'raw_protob', 'gzip_protob', 'raw_json', 'raw_sjson',
        'zip_sjson', 'asn1_json', '[raw_protob]', 'asn1_pkl']
    ignore2 = ignore.copy()
    ignore2.remove('[raw_protob]')
    # ignore_list = ['raw_protob', 'gzip_protob', '[raw_protob]', 'raw_sjson', 'compressed_sjson']
    # plot_serialization_data_as_line(result)
    # plot_serialization_data_as_line(result.loc[0.0:10, :], fig_size=(17, 7), logx=True, logy=True, lw=1.2, m_size=4)
    plot_serialization_data_as_line(
        result.loc[0.0:1, :], ignore_list=ignore, fig_size=(17, 7), logy=False,
        logx=False, max_y_values=[15, 15, None], lw=1.2, m_size=4, rolling=None)
