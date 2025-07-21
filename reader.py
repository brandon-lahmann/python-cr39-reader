import struct
import traceback

import numpy as np
import pandas as pd
from tqdm import tqdm

class ScanData:
    def __init__(self, path, frame_buffer_size=5, track_buffer_size=1000, d_bounds=(0, np.inf), e_bounds=(0, np.inf), c_bounds=(0, np.inf), a_bounds=(0, np.inf), x_bounds=(-np.inf, np.inf), y_bounds=(-np.inf, np.inf)):
        self.trailer = None
        self.frames = pd.DataFrame({
            'number': pd.Series(dtype='int32'),
            'x_position': pd.Series(dtype='float32'),
            'y_position': pd.Series(dtype='float32'),
            'num_tracks': pd.Series(dtype='int32'),
            'focus': pd.Series(dtype='float32'),
            'x_position_index': pd.Series(dtype='int32'),
            'y_position_index': pd.Series(dtype='int32'),
        })
        self.tracks = pd.DataFrame({
            'frame_number': pd.Series(dtype='int32'),
            'd': pd.Series(dtype='float32'),
            'x': pd.Series(dtype='float32'),
            'y': pd.Series(dtype='float32'),
            'e': pd.Series(dtype='int8'),
            'c': pd.Series(dtype='int8'),
            'a': pd.Series(dtype='int8'),
        })

        try:
            with open(path, 'rb') as f:
                self._f = f
                self._parse(frame_buffer_size, track_buffer_size, d_bounds, e_bounds, c_bounds, a_bounds, x_bounds, y_bounds)
        except Exception as e:
            traceback.print_exception(e)

    def _parse(self, frame_buffer_size, track_buffer_size, d_bounds, e_bounds, c_bounds, a_bounds, x_bounds, y_bounds):
        self._parse_header()
        self._parse_data(frame_buffer_size, track_buffer_size, d_bounds, e_bounds, c_bounds, a_bounds, x_bounds, y_bounds)
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

    def _parse_data(self, frame_buffer_size, track_buffer_size, d_bounds, e_bounds, c_bounds, a_bounds, x_bounds, y_bounds):
        frame_buffer = []
        track_buffer = []
        num_frames = self.header['num_x_frames'] * self.header['num_y_frames']
        try:
            for _ in tqdm(range(num_frames)):
                number = self.get_next_int()
                x_position = 1e-5 * self.get_next_int()
                y_position = 1e-5 * self.get_next_int()
                num_tracks = self.get_next_int()
                self.skip(12)
                focus = 1e-2 * self.get_next_int()
                x_position_index = self.get_next_int()
                y_position_index = self.get_next_int()

                frame_buffer.append({
                    'number': number,
                    'x_position': x_position,
                    'y_position': y_position,
                    'num_tracks': num_tracks,
                    'focus': focus,
                    'x_position_index': x_position_index,
                    'y_position_index': y_position_index
                })

                if len(frame_buffer) >= frame_buffer_size:
                    self.frames = pd.concat([self.frames, pd.DataFrame(frame_buffer)], ignore_index=True)
                    frame_buffer.clear()

                d_array = [self.get_next_short() for _ in range(num_tracks)]
                e_array = [self.get_next_byte() for _ in range(num_tracks)]
                c_array = [self.get_next_byte() for _ in range(num_tracks)]
                a_array = [self.get_next_byte() for _ in range(num_tracks)]
                x_array = [self.get_next_short() for _ in range(num_tracks)]
                y_array = [self.get_next_short() for _ in range(num_tracks)]

                for (d, e, c, a, x, y) in zip(d_array, e_array, c_array, a_array, x_array, y_array):
                    d_um = 100 * d * self.header['pixel_size']
                    x_cm = x_position - 0.5 * self.header['frame_width'] + x * self.header['pixel_size']
                    y_cm = y_position - 0.5 * self.header['frame_height'] + y * self.header['pixel_size']

                    if d_um < d_bounds[0] or d_um > d_bounds[1]:
                        continue
                    if e < e_bounds[0] or e > e_bounds[1]:
                        continue
                    if c < c_bounds[0] or c > c_bounds[1]:
                        continue
                    if a < a_bounds[0] or a > a_bounds[1]:
                        continue
                    if x_cm < x_bounds[0] or x_cm > x_bounds[1]:
                        continue
                    if y_cm < y_bounds[0] or y_cm > y_bounds[1]:
                        continue

                    track_buffer.append({'frame_number': number, 'd': d_um, 'x': x_cm, 'y': y_cm, 'e': e, 'c': c, 'a': a})

                    if len(track_buffer) >= track_buffer_size:
                        self.tracks = pd.concat([self.tracks, pd.DataFrame(track_buffer)], ignore_index=True)
                        track_buffer.clear()
        finally:
            if frame_buffer:
                self.frames = pd.concat([self.frames, pd.DataFrame(frame_buffer)], ignore_index=True)
            if track_buffer:
                self.tracks = pd.concat([self.tracks, pd.DataFrame(track_buffer)], ignore_index=True)

    def _parse_trailer(self):
        self.skip(4)
        self.trailer = self._f.read().decode('latin-1')

    def get_next_value(self, code, prefix='<'):
        size = struct.calcsize(code)
        data = self._f.read(size)
        return struct.unpack(f'{prefix}{code}', data)[0]

    def get_next_int(self):
        return self.get_next_value('i')

    def get_next_float(self):
        return self.get_next_value('f')

    def get_next_short(self):
        return self.get_next_value('h')

    def get_next_byte(self):
        return self.get_next_value('b')

    def skip(self, n):
        self._f.read(n)
