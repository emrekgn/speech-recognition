# -*- coding: utf-8 -*-

import numpy
import pyaudio
import wave
import struct


class Recorder(object):
    """
    recorder class for recording audio to a WAV file.
    Records in mono by default.
    """

    def __init__(self, channels=1, rate=44100, frames_per_buffer=1024, endpointed=False):
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self.endpointed = endpointed

    def open(self, fname, mode='wb'):
        return RecordingFile(fname, mode, self.channels, self.rate,
                             self.frames_per_buffer, self.endpointed)


class RecordingFile(object):
    def __init__(self, fname, mode, channels,
                 rate, frames_per_buffer, endpointed):
        self.fname = fname
        self.mode = mode
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self._pa = pyaudio.PyAudio()
        self.wavefile = self._prepare_file(self.fname, self.mode)
        self._stream = None
        self.endpointed = endpointed
        self.forget_factor = 1.0
        self.level = 0.0
        self.threshold = 0.0
        self.background = 0.0
        self.adjustment = 0.05
        self.is_speech = False
        self.past_is_speech = False
        self.max_energy = 0.0
        self.false_times = 0
        self.index = -5  # filter out first 5 frames
        self.condition = -1  # -1: not started, 0: 0, 1: finished

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        self.close()

    def record(self, duration):
        # Use a stream with no callback function in blocking mode
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                     channels=self.channels,
                                     rate=self.rate,
                                     input=True,
                                     frames_per_buffer=self.frames_per_buffer)
        for _ in range(int(self.rate / self.frames_per_buffer * duration)):
            audio = self._stream.read(self.frames_per_buffer)
            self.wavefile.writeframes(audio)
        return None

    def start_recording(self):
        # Use a stream with a callback in non-blocking mode
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                     channels=self.channels,
                                     rate=self.rate,
                                     input=True,
                                     frames_per_buffer=self.frames_per_buffer,
                                     stream_callback=self.callback())
        self._stream.start_stream()
        return self

    def stop_recording(self):
        self._stream.stop_stream()
        return self

    def callback(self):
        def callback(in_data, frame_count, time_info, status):
            state = pyaudio.paContinue
            if self.endpointed:
                current = []
                for i in range(0, frame_count):
                    current.append(struct.unpack('h', in_data[2 * i:2 * i + 2])[0])
                    # print current
                self.endpointing(current)
                if self.is_speech:
                    if self.condition == -1:
                        self.condition = 0
                    self.wavefile.writeframes(in_data)
                else:
                    if self.condition == 0:
                        self.condition = 1
            if self.condition == 1:
                state = pyaudio.paComplete
            return in_data, state

        return callback


    def close(self):
        self._stream.close()
        self._pa.terminate()
        self.wavefile.close()

    def _prepare_file(self, fname, mode='wb'):
        wavefile = wave.open(fname, mode)
        wavefile.setnchannels(self.channels)
        wavefile.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(self.rate)
        return wavefile

    def endpointing(self, current):
        if self.false_times > 40:
            self.is_speech = False
        energy = 0.0
        for i in range(len(current)):
            energy += pow(current[i], 2)
        energy = 10 * numpy.log(energy)
        print self.threshold, self.level, self.background
        if self.index == 0:
            self.level = energy
        if self.index < 10 and self.index >= 0:
            if self.max_energy < energy:
                self.max_energy = energy
            self.background += energy
            if self.index == 9:
                self.background /= 10
                self.threshold = self.max_energy / 2
        if self.index >= 10:
            if energy < self.background:
                self.background = energy
            else:
                self.background += (energy - self.background) * self.adjustment
            self.level = (self.level * self.forget_factor + energy) / (self.forget_factor + 1)
            if self.level < self.background:
                self.level = self.background
            if self.level - self.background > self.threshold:
                self.is_speech = True
                self.false_times = 0
            else:
                self.false_times += 1
            if self.is_speech != self.past_is_speech:
                if self.is_speech:
                    print "speech begin at %d\n" % self.index
                    print 'energy = %d\n' % energy
                else:
                    print "speech end at %d\n" % self.index
                    print 'energy = %d\n' % energy
            self.past_is_speech = self.is_speech
        self.index += 1
