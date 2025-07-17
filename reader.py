import struct
import pandas as pd
from tqdm import tqdm

class ScanData:
    def __init__(self, path):
        self.index = 0
        with open(path, 'rb') as f:
            self.data = f.read()
        self._parse()

    def _parse(self):
        self._parse_header()
        self._parse_data()
        self._parse_trailer()

    def _parse_header(self):
        self.header = {
            'version_number': self.get_next_int(),
            'num_x_frames': self.get_next_int(),
            'num_y_frames': self.get_next_int(),
            'num_bins': self.get_next_int(),
            'pixel_size': 1e-4 * self.get_next_float(),
            'pixels_per_bin': self.get_next_float(),
            'border_limit': self.get_next_int(),
            'contrast_limit': self.get_next_int(),
            'eccentricity_limit': self.get_next_int(),
            'M': self.get_next_int(),
            'frame_width': self.get_next_int(),
            'frame_height': self.get_next_int(),
        }
        self.header['frame_width'] *= self.header['pixel_size']
        self.header['frame_height'] *= self.header['pixel_size']

    def _parse_data(self):
        frame_data = []
        track_data = []
        num_frames = self.header['num_x_frames'] * self.header['num_y_frames']
        for _ in tqdm(range(num_frames)):
            number = self.get_next_int()
            x_position = 1e-5 * self.get_next_int()
            y_position = 1e-5 * self.get_next_int()
            num_tracks = self.get_next_int()
            self.skip(12)
            focus = 1e-2 * self.get_next_int()
            x_position_index = self.get_next_int()
            y_position_index = self.get_next_int()

            frame_data.append({
                'number': number,
                'x_position': x_position,
                'y_position': y_position,
                'num_tracks': num_tracks,
                'focus': focus,
                'x_position_index': x_position_index,
                'y_position_index': y_position_index
            })

            d_array = [self.get_next_short() for _ in range(num_tracks)]
            e_array = [self.get_next_byte() for _ in range(num_tracks)]
            c_array = [self.get_next_byte() for _ in range(num_tracks)]
            a_array = [self.get_next_byte() for _ in range(num_tracks)]
            x_array = [self.get_next_short() for _ in range(num_tracks)]
            y_array = [self.get_next_short() for _ in range(num_tracks)]

            for (d, e, c, a, x, y) in zip(d_array, e_array, c_array, a_array, x_array, y_array):
                track_data.append({
                    'frame_number': number,
                    'd': 100 * d * self.header['pixel_size'],
                    'x': x_position - 0.5 * self.header['frame_width'] + x * self.header['pixel_size'],
                    'y': y_position - 0.5 * self.header['frame_height'] + y * self.header['pixel_size'],
                    'e': e,
                    'c': c,
                    'a': a
                })

        self.frames = pd.DataFrame(frame_data)
        self.tracks = pd.DataFrame(track_data)

    def _parse_trailer(self):
        self.skip(4)
        self.trailer = self.data[self.index::].decode('latin-1')

    def get_next_value(self, code, prefix='<'):
        size = struct.calcsize(code)
        # print(size, self.index, self.data[self.index:(self.index+size)])
        value = struct.unpack(f'{prefix}{code}', self.data[self.index:(self.index+size)])[0]
        self.index += size
        return value

    def get_next_int(self):
        return self.get_next_value('i')

    def get_next_float(self):
        return self.get_next_value('f')

    def get_next_short(self):
        return self.get_next_value('h')

    def get_next_byte(self):
        return self.get_next_value('b')

    def skip(self, n):
        self.index += n
