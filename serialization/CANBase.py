# -*- coding: utf-8 -*-
"""
:version: 1.0
:copyright: 2018, Tim Häberlein, TU Dresden, FZM
"""

# -------- start import block ---------
import cantools
from cantools import database
import time
import logging
import datetime
# -------- end import block ---------


class CANBase:
    """
    A class that holds all basic CAN methods **(optimised for python 3.6/3.7)**
    """

    @staticmethod
    def get_pcan_data(frame_data, frame_length) -> list:
        """
        Convert frame_data for a pcan_msg.

        :param frame_data: hex string of frame_data
        :param frame_length: DLC of the frame_data
        :type frame_data: str
        :type frame_length: int

        :return: pcan_data
        :rtype: list
        """

        pcan_data = list()
        for i in range(frame_length):
            pcan_data.append(int(frame_data[i*2:i*2+2], 16))

        return pcan_data

    @staticmethod
    def get_frame_data(pcan_data) -> str:
        """
        Convert frame_data from pcan_msg.

        :param pcan_data: data_array of a pcan_msg
        :type pcan_data: bytearray
        :return: frame_data as hex
        :rtype: str
        """

        frame_data = ''
        for data in pcan_data:
            frame_data += '%0.2x' % data

        return frame_data

    @staticmethod
    def get_frame_type(pcan_type) -> str:
        """
        Convert a frame type from pcan type to string:

        - **'EXT'** for a Extended Frame
        - **'STD'** for a Standard Frame

        :param pcan_type: type from pcan_msg
        :type pcan_type: int
        :raise ValueError: if pcan_type is unknown
        :return: frame_type
        :rtype: str
        """

        if pcan_type == 0x00:
            return 'STD'
        elif pcan_type == 0x02:
            return 'EXT'
        else:
            raise ValueError('pcan_type is unknown')

    @staticmethod
    def get_pcan_type(frame_type) -> int:
        """
        Convert a frame type from string to pcan_type:

        - **'EXT'** for a Extended Frame: 0x02
        - **'STD'** for a Standard Frame: 0x00

        :param frame_type: frame_type
        :type frame_type: str
        :raise ValueError: if frame_type is unknown
        :return: pcan_type
        :rtype: int
        """

        if frame_type == 'STD':
            return 0x00
        elif frame_type == 'EXT':
            return 0x02
        else:
            raise ValueError('frame_type is unknown')

    @staticmethod
    def id_2_str(pcan_id) -> str:
        """
        Method to convert frame_id from int 2 str in hex

        :param pcan_id: id from a pcan_msg
        :type pcan_id: int
        :return: frame_id as hex
        :rtype: str
        """

        # return '%0.2x' % pcan_id
        return hex(pcan_id)

    @staticmethod
    def id_2_int(frame_id) -> int:
        """
        Convert a frame_id from string in hex to int

        :param frame_id: Frame ID as string in hex
        :type frame_id: str
        :return: frame_id
        :rtype: int
        """

        return int(frame_id, 16)

    @staticmethod
    def get_time(start_ts, start_pcan_ts, pcan_ts) -> float:
        """
        Method to calc an timestamp for a PCAN_Timestamp in ms.

        :param start_ts: unix-timestamp from start can-reading
        :param start_pcan_ts: pcan_ts from start can-reading
        :param pcan_ts: actual ts from can-reading
        :return: ts im ms
        :rtype: int
        """

        # actual ts = start_ts + (actual pcan_ts[sec] - start_pcan_ts[sec]
        ts = start_ts + (CANBase.pcan_ts_2_sec(pcan_ts) - CANBase.pcan_ts_2_sec(start_pcan_ts))
        # convert in milliseconds
        # ts = int(round(ts*10**3))
        return ts

    @staticmethod
    def pcan_ts_2_sec(pcan_ts) -> float:
        """
        Method to calc pcan_ts in seconds

        :param pcan_ts: timestamp from can_read
        :type pcan_ts: TPCANTimestamp
        :return: seconds from pcan_ts
        :rtype: float
        """
        micro_seconds = pcan_ts.micros + 1000 * pcan_ts.millis + 0x100000000 * 1000 * pcan_ts.millis_overflow
        return float(micro_seconds*10**-6)

    @staticmethod
    def load_dbc_obj(path='c:/users/s7285521/ideaprojects/python-cocapl_car/static/dbc/sample.dbc'):
        """
        Load a dbc-file into a cantools dbc obj

        :param path: path + file to dbc
        :type path: str
        :return: dbc_obj
        """
        with open(path) as dbc_file:
            dbc_obj = cantools.database.load(dbc_file)
        return dbc_obj


# ------------------------------------------------------------------------------------------------------------------
# CAN Frame
# ------------------------------------------------------------------------------------------------------------------
class CocaplCanFrame(CANBase):
    """
    A class that represents a CoCaPl CAN Object
    """

    def __init__(self, frame_id='0x600', length=8,
                 frame_type='STD', data='0100000001000000',
                 ts=None, vin='TESTOBUXX2019XX01', ch='s1_can/send', raw_data=None):
        """
        Initialize a CoCaPl CAN Object:

        raw_data looks like this:
            - **frame_id** (str): can_frame id in hex
            - **length** (int): DLC (length) of the frame, number of bytes
            - **frame_type** (str):
            - **data** (str): raw data of the frame in hex
            - **ts** (str): unix-timestamp in seconds
            - **ch**
            - **vin**

        .. note:: if you define raw_data you will override all other values

        :param frame_id: can_frame id in hex
        :type frame_id: str
        :param length: DLC (length) of the frame, number of bytes
        :type length: int
        :param frame_type: type of the frame ('STD' or 'EXT')
        :type frame_type: str
        :param data: raw data of the frame in hex
        :type data: str
        :param ts: unix-timestamp in seconds
        :type ts: float
        :param vin: vehicle identification number
        :type vin: str
        :param ch: name of data channel
        :type ch: str
        :param raw_data: raw data of the frame as dict **overwrites all other values**
        :type raw_data: dict
        """
        if ts is None:
            ts = time.time()
        self.__frame_id = frame_id
        self.__length = length
        self.__frame_type = frame_type
        self.__data = data
        self.__ts = ts
        self.__vin = vin
        self.__ch = ch
        self.__raw_data = dict()
        if raw_data is None:
            # if raw_data is not set get values from properties
            self.__update_raw_data()
        else:
            # if raw_data is set -> set self._raw_data & update properties
            self.__raw_data = raw_data
            self.__update_properties()

        self.__logger = logging.getLogger(__name__)

    def encode_frame(self, can_db, signal_list):
        """
        Encode signals to this can frame.

        .. note: you will overwrite the actual data field

        :param can_db: Database object with signal identificated by name & frame_id
        :type can_db: cantools.database.Database
        :param signal_list: list of CoCaplSignals to append
        :type signal_list: list
        :return: -
        """
        message: cantools.database.Message = can_db.get_message_by_frame_id(int(self.frame_id, 16))

        # encode all other values with 0
        frame_signals = message.signals
        signal_dict = dict()
        # build signal dict
        for frame_signal in frame_signals:
            signal_dict[frame_signal.name] = 0

        for signal in signal_list:
            signal_dict[signal.signal_name] = signal.value if signal.signal_name in signal_dict.keys() else 0

        self.data = message.encode(signal_dict).hex()

    def decode_frame(self, can_db):
        """
        Decode the can-frame to a list of CocaplSignals

        :param can_db: cantools Database object from can_db
        :type can_db: cantools.database.Database
        :return: CoCaplSignals from can_frame
        :rtype: list
        """

        signal_list = list()
        try:
            message: cantools.database.Message = can_db.get_message_by_frame_id(int(self.frame_id, 16))
            signals = message.decode(data=bytes.fromhex(self.data))
            for key, value in signals.items():
                signal_list.append(CocaplSignal(
                    frame_id=self.frame_id, signal_name=key, value=value, unit=message.get_signal_by_name(key).unit,
                    signal_comment=message.get_signal_by_name(key).comment, ts=self.ts, vin=self.vin, ch=self.ch
                ))

        except KeyError:
            self.__logger.warning('message not found by id in given dbc-file')
            signal_list.append(CocaplSignal(
                frame_id=self.frame_id, signal_name='unknown_{id}'.format(id=self.frame_id),
                value=self.data, ts=self.ts, vin=self.vin, ch=self.ch
            ))

        return signal_list

    # ------------------------------------------
    # properties
    # ------------------------------------------
    @property
    def raw_data(self) -> dict:
        """
        Get the internal frame_dict:

        - **id** (str): can_frame id in hex
        - **length** (int): DLC (length) of the frame, number of bytes
        - **type** (str): type of the frame ('STD' or 'EXT')
        - **data** (str): row data of the frame in hex
        - **time** (str): unix-timestamp in ms (e.g. 1531960033123 = 19-Jul-18 00:27:13,123)
        :return: frame_dict
        :rtype: dict
        """
        self.__update_raw_data()
        return self.__raw_data

    @raw_data.setter
    def raw_data(self, value):
        for key in value.keys():
            if key not in self.__raw_data.keys():
                raise ValueError("dict contains not the correct keys")
            self.__raw_data = value
            self.__update_properties()

    @property
    def frame_id(self):
        return self.__frame_id

    @frame_id.setter
    def frame_id(self, value):
        if not (type(value) is str):
            raise ValueError("ID must be string and starts with 0x")
        elif not (value.startswith('0x')):
            raise ValueError("ID must be string and starts with 0x")
        self.__frame_id = value
        # self.__update_frame_dict()

    @property
    def length(self):
        return self.__length

    @length.setter
    def length(self, value):
        if not (type(value) is int):
            raise ValueError("LENGTH must be int and between 1 and 8")
        if (value > 8) and (value < 1):
            raise ValueError("LENGTH must be int and between 1 and 8")
        self.__length = value
        # self.__update_frame_dict()

    @property
    def data(self) -> str:
        """
        Get the raw data as string
        :return: data
        :rtype: str
        """
        return self.__data

    @data.setter
    def data(self, value):
        """
        Set the frame raw data as string betwwen '00' and 'FFFFFFFFFFFFFFFF'
        :param value:
        :return:
        """
        if not (type(value) is str):
            raise ValueError("DATA must be str and between 00 and FFFFFFFFFFFFFFFF")
        length = int(len(value)/2)
        pcan_data = CANBase.get_pcan_data(value, length)
        for data in pcan_data:
            if data > 255 or data < 0:
                raise ValueError("DATA must be str and between 00 and FFFFFFFFFFFFFFFF")
        self.__length = length
        self.__data = value
        # self.__update_frame_dict()

    @property
    def frame_type(self):
        return self.__frame_type

    @frame_type.setter
    def frame_type(self, value):
        if not (value == 'STD' or value == 'EXT'):
            raise ValueError('TYPE must be STD or EXT')
        self.__frame_type = value
        # self.__update_frame_dict()

    @property
    def ts(self):
        return self.__ts

    @ts.setter
    def ts(self, value):
        if type(value) is not float:
            raise ValueError('TS must be float')
        self.__ts = value

    @property
    def dt_ts(self):
        """
        Returns the actual time as datetime object
        :return: datetime object of time
        :rtype: datetime.datetime
        """
        return datetime.datetime.fromtimestamp(self.__ts)

    @property
    def ch(self):
        return self.__ch

    @ch.setter
    def ch(self, value):
        if type(value) is not str:
            raise ValueError('Ch must be str')
        self.__ch = value

    @property
    def vin(self):
        return self.__vin

    @vin.setter
    def vin(self, value):
        if type(value) is not str:
            raise ValueError('VIN must be str')
        self.__vin = value

    # ------------------------------------------
    # private methods
    # ------------------------------------------
    def __update_raw_data(self):
        frame_dict = dict()
        frame_dict['frame_id'] = self.__frame_id
        frame_dict['length'] = self.__length
        frame_dict['frame_type'] = self.__frame_type
        frame_dict['data'] = self.__data
        frame_dict['ts'] = self.__ts
        frame_dict['ch'] = self.__ch
        frame_dict['vin'] = self.__vin
        self.__raw_data = frame_dict

    def __update_properties(self):
        data = self.__raw_data.copy()
        self.frame_id = data['frame_id']
        self.length = data['length']
        self.frame_type = data['frame_type']
        self.data = data['data']
        self.ts = data['ts']
        try:
            self.ch = data['ch']
        except KeyError:
            pass
        try:
            self.vin = data['vin']
        except KeyError:
            pass

        del data

    def __repr__(self):
        return 'cocaplcanframe("{frame_id}": {data}, {vin}, {ch})'.format(
            frame_id=self.frame_id,
            data=self.data,
            vin=self.vin,
            ch=self.ch
        )


# ------------------------------------------------------------------------------------------------------------------
# Signal
# ------------------------------------------------------------------------------------------------------------------
class CocaplSignal(CANBase):
    """
    A class that represents a CoCaPl Signal
    """
    def __init__(self, frame_id='0x0', signal_name=None,
                 value=None, unit=None,
                 signal_comment=None,
                 ts=time.time(), vin='default', ch='s1_can/send', raw_data=None):
        """
        Initialize a CoCaPl Signal Object

        :param frame_id: can_frame id in hex from can_message **default='0x0'**
        :type frame_id: str
        :param signal_name: name of the signal
        :type signal_name: str
        :param value: value of the signal
        :type value: any
        :param unit: unit of the signal, None if no unit
        :type unit: str
        :param signal_comment: comment for the signal, None if no comment
        :type signal_comment: str
        :param ts: unix-timestamp in seconds
        :type ts: float
        :param vin: vehicle identification number
        :type vin: str
        :param ch: name of data channel
        :type ch: str
        :param raw_data: Raw data of the signal as dict
        :type raw_data: dict
        """

        self.__ts = ts
        self.__vin = vin
        self.__signal_name = signal_name
        self.__value = value
        self.__unit = unit
        self.__frame_id = frame_id
        self.__signal_comment = signal_comment
        self.__ch = ch
        self.__raw_data = dict()
        if raw_data is None:
            # if raw data is none get data from internal values
            self.__update_raw_data()
        else:
            # if raw data is not None set raw_data & properties
            self.__raw_data = raw_data
            self.__update_properties()

        self.__logger = logging.getLogger(__name__)

    # ------------------------------------------
    # methods
    # ------------------------------------------
    def decode_2_frame(self, can_db, signal_list=None):
        """
        Decode the CocaplSignal to a CocoplCanFrame.

        :param can_db: Database object with signal identificated by name & frame_id
        :type can_db: cantools.database.Database
        :param signal_list: list of CoCaplSignals to append
        :type signal_list: list
        :return: the encoded can-frame
        :rtype: CocaplCanFrame
        """

        if signal_list is None:
            signal_list = []

        signal_list.append(self)

        message: cantools.database.Message = can_db.get_message_by_frame_id(int(self.frame_id, 16))

        # encode all other values with 0
        signals = message.signals
        signal_dict = dict()
        for signal in signals:
            found = False
            for sig in signal_list:
                if signal.name == sig.signal_name:
                    found = True
                    signal_dict[signal.name] = sig.value
                else:
                    pass
            # if no signal is found in signal_list write zero
            if not found:
                signal_dict[signal.name] = 0
                print('not found')

        data = message.encode(signal_dict).hex()
        return CocaplCanFrame(
            frame_id=self.frame_id, length=message.length, frame_type='EXT' if message.is_extended_frame else 'STD',
            data=data, ts=self.ts, vin=self.vin, ch=self.ch
        )
    # ------------------------------------------
    # properties
    # ------------------------------------------
    @property
    def dt_ts(self) -> datetime.datetime:
        """
        Returns the actual time as datetime object
        :return: datetime object of time
        :rtype: datetime.datetime
        """
        return datetime.datetime.fromtimestamp(self.__ts)

    @property
    def raw_data(self) -> dict:
        return self.__raw_data

    @raw_data.getter
    def raw_data(self):
        """
        Get the internal signal_dict:
            - **ts** (float): unix-timestamp in seconds
            - **vin** (str): vehicle identification number
            - **signal_name** (str): name of the signal
            - **value** (Any): value of the signal
            - **unit** (str): unit of the signal, None if no comment
            - **frame_id** (int): id for the frame with the signal from dbc-file
            - **signal_comment** (str): comment for the signal
            - **channel** (str): mqtt-channel

        :return: frame_dict
        :rtype: dict
        """
        self.__update_raw_data()
        return self.__raw_data

    @raw_data.setter
    def raw_data(self, value):
        for key in value.keys():
            if key not in self.__raw_data.keys():
                raise ValueError("dict contains not the correct keys")
            self.__raw_data = value
            self.__update_properties()

    @property
    def frame_id(self):
        return self.__frame_id

    @frame_id.setter
    def frame_id(self, value):
        if not (type(value) is str):
            raise ValueError("ID must be string and starts with 0x")
        elif not (value.startswith('0x')):
            raise ValueError("ID must be string and starts with 0x")
        self.__frame_id = value

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        self.__value = value

    @property
    def ts(self):
        return self.__ts

    @ts.setter
    def ts(self, value):
        if type(value) is not float:
            raise ValueError('TS must be float')
        self.__ts = value

    @property
    def ch(self):
        return self.__ch

    @ch.setter
    def ch(self, value):
        if type(value) is not str:
            raise ValueError('CH must be str')
        self.__ch = value

    @property
    def vin(self):
        return self.__vin

    @vin.setter
    def vin(self, value):
        if type(value) is not str:
            raise ValueError('VIN must be str')
        self.__vin = value

    @property
    def signal_name(self):
        return self.__signal_name

    @signal_name.setter
    def signal_name(self, value):
        if type(value) is not str:
            raise ValueError('SIGNAL NAME must be str')
        self.__signal_name = value

    @property
    def signal_comment(self):
        return self.__signal_comment

    @signal_comment.setter
    def signal_comment(self, value):
        if type(value) is not str:
            raise ValueError('SIGNAL COMMENT must be str')
        self.__signal_comment = value

    @property
    def unit(self):
        return self.__unit

    @unit.setter
    def unit(self, value):
        if type(value) is not str:
            raise ValueError('UNIT must be str')
        self.__unit = value

    # ------------------------------------------
    # private methods
    # ------------------------------------------
    def __update_raw_data(self):
        self.__raw_data = {
            'frame_id': self.__frame_id,
            'ts': self.__ts,
            'vin': self.__vin,
            'signal_name': self.__signal_name,
            'value': self.__value,
            'unit': self.__signal_comment,
            'signal_comment': self.__signal_comment,
            'ch': self.__ch,
        }

    def __update_properties(self):

        data = self.__raw_data.copy()
        self.ch = data['ch']
        self.signal_comment = data['signal_comment']
        self.unit = data['unit']
        self.value = data['value']
        self.signal_name = data['signal_comment']
        self.vin = data['vin']
        self.ts = data['ts']
        self.frame_id = data['frame_id']
        del data

    def __repr__(self):
        return "cocaplsignal('{name}': {value}, {vin}, {ch}".format(
            name=self.signal_name,
            value=self.value,
            vin=self.vin,
            ch=self.ch
        )


if __name__ == '__main__':
    frame = CocaplCanFrame()
    dbc_file = frame.load_dbc_obj()
    sig_list = frame.decode_frame(dbc_file)
    frame2 = CocaplCanFrame()
    frame2.encode_frame(dbc_file, sig_list)
