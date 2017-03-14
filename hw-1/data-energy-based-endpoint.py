# -*- coding: utf-8 -*-

import numpy
import pyaudio
import wave
import struct


class RecordingFile(object):

    def __init__(self, filename, mode, channels=1, rate=44100, frames_per_buffer=1024):
        self.filename = filename
        self.mode = mode
        self.channels = channels
        self.rate = rate
        self.frames_per_buffer = frames_per_buffer
        self._pa = pyaudio.PyAudio()
        self.wavefile = self._prepare_file(self.filename, self.mode)
        self._stream = None
        self.forget_factor = 1.0
        self.level = 0.0
        self.threshold = 0.0
        self.background = 0.0
        self.adjustment = 0.05
        self.num_silent = 0
        self.snd_started = False
        self.is_speech = False
        self.prev_is_speech = False
        self.max_energy = 0.0
        self.index = 0.0
        self.begin_index = 0
        self.end_index = 0

    def __enter__(self):
        return self

    def __exit__(self, exception, value, traceback):
        self.close()

    def start_recording(self):
        # Use a stream with a callback in non-blocking mode
        print(1)
        self._stream = self._pa.open(format=pyaudio.paInt16,
                                     channels=self.channels,
                                     rate=self.rate,
                                     input=True,
                                     frames_per_buffer=self.frames_per_buffer,
                                     stream_callback=self.callback())
        print(2)
        self._stream.start_stream()
        return self

    def stop_recording(self):
        self._stream.stop_stream()
        return self

    def callback(self):
        print(3)
        def callback(in_data, frame_count, time_info, status):
            print(4)
            current = []
            for i in range(0, frame_count):
                current.append(struct.unpack('h', in_data[2 * i:2 * i + 2])[0])
                print(current)
            self.endpointing(current)
            if self.is_speech:
                self.wavefile.writeframes(in_data)
                if not self.snd_started:
                    self.snd_started = True
            elif not self.is_speech and self.snd_started:
                self.num_silent += 1
            elif self.snd_started and self.num_silent > 20:
                print("End of speech detected. Stopping recording...")
                return None, pyaudio.paComplete
            return None, pyaudio.paContinue

        return callback

    def close(self):
        self._stream.close()
        self._pa.terminate()
        self.wavefile.close()

    def _prepare_file(self, filename, mode='wb'):
        wavefile = wave.open(filename, mode)
        wavefile.setnchannels(self.channels)
        wavefile.setsampwidth(self._pa.get_sample_size(pyaudio.paInt16))
        wavefile.setframerate(self.rate)
        return wavefile

    def endpointing(self, current):
        energy = 0.0
        # print current
        # print len(current)
        self.is_speech = False
        for i in range(len(current)):
            energy += pow(current[i], 2)
        energy = 10 * numpy.log(energy)
        # print self.threshold, self.level, self.background
        # print energy
        if self.index == 0:
            self.level = energy
        # threshold equals to max energy of first ten frames divide 4
        if self.index < 10 and self.index >= 0:
            if self.max_energy < energy:
                self.max_energy = energy
            self.background += energy
            if self.index == 9:
                self.background /= 10
                self.threshold = self.max_energy / 8
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
            if self.is_speech != self.prev_is_speech:
                if self.is_speech:
                    self.begin_index = self.index
                    # print "speech begin at %d\n" % self.begin_index
                    # print 'energy = %d\n' % energy
                else:
                    self.end_index = self.index
            self.prev_is_speech = self.is_speech
        self.index += 1

if __name__ == '__main__':
    rec = RecordingFile('energy-based-recording.wav', 'wb', )
    input("Press Enter to begin recording...")
    rec.start_recording()
    print("Done - result written to energy-based-recording.wav")
