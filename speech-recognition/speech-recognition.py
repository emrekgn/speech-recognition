#!/usr/bin/env python

from __future__ import division
from sys import byteorder
from array import array
from tkinter import *
from tkinter import messagebox
from tkinter.ttk import *
import os
from python_speech_features import mfcc
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy
import pyaudio
import wave


class SpeechRecognition():
    def __init__(self):
        # Common parameters
        self.CHUNK_SIZE = 1024  # num of frames to read at a time
        self.FORMAT = pyaudio.paInt16
        self.FREQUENCY = 44100
        self.FRAME_SIZE_IN_MS = 0.01
        self.NUM_SAMPLES_PER_FRAME = int(self.FREQUENCY * self.FRAME_SIZE_IN_MS)
        self.ENERGY_THRESHOLD = 50
        self.SILENCE_THRESHOLD = 500
        self.MAX_SILENCE_COUNT = 10
        self.DATA_PATH = './data'
        # Output paths and file names
        self.OUTPUT_WAV = 'output/speech-endpoint.wav'
        self.OUTPUT_ZERO_CROSSING_RATE = 'output/zero-crossing-rate.png'
        self.OUTPUT_ENERGY = 'output/short-time-energy.png'
        self.OUTPUT_SPECTROGRAM = 'output/spectrogram.png'
        self.OUTPUT_RAW_SIGNAL = 'output/raw-signal.png'
        self.OUTPUT_MFCC_TXT = 'output/mfcc-features.txt'
        self.OUTPUT_MFCC_FIG = 'output/mfcc-features.png'
        # Tkinter GUI
        self.root = Tk()
        self.var = IntVar()
        self.var.set(1)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)


    def main(self):
        self.root.title = 'Speech Recognition'

        left_frame = Frame(self.root, width=180)
        left_frame.pack(side=LEFT)

        self.log = Text(left_frame, bg='white')
        self.log.pack()

        right_frame = Frame(self.root)
        right_frame.pack(side=RIGHT)

        Radiobutton(right_frame, text='Record from mic & analyze', variable=self.var, value=1).pack(side=TOP, anchor=W, padx=10, pady=10)
        Radiobutton(right_frame, text='Select one of the existing audio files & analyze', variable=self.var,
                    value=2).pack(side=TOP, anchor=W, padx=10, pady=10)

        scrollbar = Scrollbar(right_frame)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.existing_audio_files = Listbox(right_frame, yscrollcommand=scrollbar.set)
        for index, name in enumerate(os.listdir(self.DATA_PATH)):
            if os.path.isfile(os.path.join(self.DATA_PATH, name)):
                self.existing_audio_files.insert(index, name)

        self.existing_audio_files.pack(side=TOP, fill=BOTH, padx=10, pady=10)
        scrollbar.config(command=self.existing_audio_files.yview)

        start_button = Button(right_frame, text='Start!', command=self.start)
        start_button.pack(side=TOP, padx=10, pady=10)

        self.root.mainloop()

    def start(self):
        self.log.insert(END, 'Starting...\n')
        if self.var.get() == 1:
            # Record from mic & analyze
            self.FREQUENCY, signal = self.record_from_mic()
            self.analyze(signal)
        else:
            # Read from file & analyze
            selection = self.existing_audio_files.curselection()
            self.FREQUENCY, signal = wavfile.read(os.path.join(self.DATA_PATH, self.existing_audio_files.get(selection[0])))
            self.log.insert(END, 'Reading existing audio file...\n')
            self.analyze(signal)

    def analyze(self, signal):
        self.log.insert(END, '\nAnalyzing audio signal...\n')
        signal = signal / max(abs(signal))  # scale for plotting and calculations
        assert min(signal) >= -1 and max(signal) <= 1

        # Prints some stats
        self.log.insert(END, 'Frequency ==> {} Hz\n'.format(self.FREQUENCY))  # sampling rate
        self.log.insert(END, 'Length of signal  ==> {} samples\n'.format(len(signal)))
        self.log.insert(END, 'Signal  ==> {}\n'.format(signal))

        sampsPerMilli = int(self.FREQUENCY / 1000)
        millisPerFrame = 20
        sampsPerFrame = sampsPerMilli * millisPerFrame
        nFrames = int(len(signal) / sampsPerFrame)  # number of non-overlapping _full_ frames

        self.log.insert(END, 'Samples/millisecond  ==> {}\n'.format(sampsPerMilli))
        self.log.insert(END, 'Samples/[%dms]frame  ==> % {} {}\n'.format(millisPerFrame, sampsPerFrame))
        self.log.insert(END, 'Number of frames     ==> {}\n'.format(nFrames))

        # Raw signal
        plt.figure()
        plt.plot(signal)
        plt.title('Raw Signal')
        plt.xlabel('Sample')
        plt.autoscale(tight='both')
        plt.savefig(self.OUTPUT_RAW_SIGNAL)

        '''
        # Spectrogram
        N = len(signal)  # total number of signals
        curPos = 0
        Win = round(self.FREQUENCY * 0.040)
        Step = round(self.FREQUENCY * 0.040)
        countFrames = 0
        nfft = int(Win / 2)
        specgram = numpy.array([], dtype=numpy.float64)

        while (curPos + Win - 1 < N):
            countFrames += 1
            x = signal[curPos:curPos + Win]
            curPos = curPos + Step
            X = abs(fft(x))
            X = X[0:nfft]
            X = X / len(X)

            if countFrames == 1:
                specgram = X ** 2
            else:
                specgram = numpy.vstack((specgram, X))

        fig, ax = plt.subplots()
        imgplot = plt.imshow(specgram.transpose()[::-1, :])
        Fstep = int(nfft / 5.0)
        FreqTicks = range(0, int(nfft) + Fstep, Fstep)
        FreqTicksLabels = [str(self.FREQUENCY / 2 - int((f * self.FREQUENCY) / (2 * nfft))) for f in FreqTicks]
        ax.set_yticks(FreqTicks)
        ax.set_yticklabels(FreqTicksLabels)
        TStep = round(countFrames / 3)
        TimeTicks = range(0, countFrames, TStep)
        TimeTicksLabels = ['%.2f' % (float(t * Step) / self.FREQUENCY) for t in TimeTicks]
        ax.set_xticks(TimeTicks)
        ax.set_xticklabels(TimeTicksLabels)
        ax.set_xlabel('time (secs)')
        ax.set_ylabel('freq (Hz)')
        imgplot.set_cmap('jet')
        plt.colorbar()
        plt.savefig(self.OUTPUT_SPECTROGRAM)
        '''

        # Short-time energy
        STEs = []
        for k in range(nFrames):
            startIdx = k * sampsPerFrame
            stopIdx = startIdx + sampsPerFrame
            window = numpy.zeros(signal.shape)
            window[startIdx:stopIdx] = 1  # rectangular window
            STE = sum((signal ** 2) * (window ** 2))
            STEs.append(STE)

        plt.figure()
        plt.plot(STEs)
        plt.title('Energy')
        plt.ylabel('ENERGY')
        plt.xlabel('FRAME')
        plt.autoscale(tight='both')
        plt.savefig(self.OUTPUT_ENERGY)

        # Zero-crossing rate
        DC = numpy.mean(signal)
        newSignal = signal - DC  # create a new signal, preserving old
        self.log.insert(END, 'DC               ==> {}\n'.format(DC))
        self.log.insert(END, 'mean(newSignal)  ==> {}\n'.format(numpy.mean(newSignal)))
        ZCCs = []  # list of short-time zero crossing counts
        for i in range(nFrames):
            startIdx = i * sampsPerFrame
            stopIdx = startIdx + sampsPerFrame
            s = newSignal[startIdx:stopIdx]  # /s/ is the frame, named to correspond to the equation
            ZCC = 0
            for k in range(1, len(s)):
                ZCC += 0.5 * abs(numpy.sign(s[k]) - numpy.sign(s[k - 1]))
            ZCCs.append(ZCC)

        plt.figure()
        plt.plot(ZCCs)
        plt.title('Zero Crossing Rate')
        plt.ylabel('ZCC')
        plt.xlabel('FRAME')
        plt.autoscale(tight='both')
        plt.savefig(self.OUTPUT_ZERO_CROSSING_RATE)

        # Extract features
        mfcc_features = mfcc(signal, self.FREQUENCY, nfilt=40, lowfreq=50)
        numpy.savetxt(self.OUTPUT_MFCC_TXT, mfcc_features)
        self.log.insert(END, '\nMFCC:\nNumber of windows = {}\n'.format(mfcc_features.shape[0]))
        self.log.insert(END, 'Length of each feature = {}\n'.format(mfcc_features.shape[1]))

        # plt.figure()
        # Transform the matrix so that the time domain is horizontal
        mfcc_features = mfcc_features.T
        plt.matshow(mfcc_features)
        plt.title('MFCC')
        plt.savefig(self.OUTPUT_MFCC_FIG)
        plt.show()

        self.log.insert(END, "\nDone - results written to output directory.\n")

    def record_from_mic(self):
        """

        :return:
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT, channels=1, rate=self.FREQUENCY,
                        input=True, output=True,
                        frames_per_buffer=self.CHUNK_SIZE)
        sample_width = p.get_sample_size(self.FORMAT)
        self.log.insert(END, '\nRecording from microphone with automatic endpointing...\n')

        num_silence = 0
        snd_started = False

        signal = array('h')
        while 1:
            # little endian, signed short
            chunk = array('h', stream.read(self.CHUNK_SIZE))
            if byteorder == 'big':
                chunk.byteswap()
            signal.extend(chunk)

            silent = self.is_silent(chunk)

            if silent and snd_started:
                num_silence += 1
            elif not silent and not snd_started:
                snd_started = True
            if snd_started and num_silence > self.MAX_SILENCE_COUNT:
                self.log.insert(END, "End of speech detected. Stopping recording...\n")
                break

        stream.stop_stream()
        stream.close()
        p.terminate()
        self.log.insert(END, 'Stopped recording.\n')

        signal = self.normalize(signal)
        signal = self.trim(signal)
        #signal = self.add_silence(signal, 0.5)
        self.write_to_file(signal, sample_width, self.OUTPUT_WAV)
        signal = numpy.fromiter(signal, dtype=numpy.int16)
        return self.FREQUENCY, signal

    def normalize(self, signal):
        """
        Average the volume out
        :param signal:
        :return:
        """
        MAXIMUM = 16384
        times = float(MAXIMUM) / max(abs(i) for i in signal)

        new_signal = array('h')
        for i in signal:
            new_signal.append(int(i * times))
        return new_signal

    def trim(self, signal):
        """
        Trim the blank spots at the start and end
        :param signal:
        :return:
        """

        def _trim(snd_data):
            snd_started = False
            new_signal = array('h')

            for i in snd_data:
                if not snd_started and abs(i) > self.SILENCE_THRESHOLD:
                    snd_started = True
                    new_signal.append(i)
                elif snd_started:
                    new_signal.append(i)
            return new_signal

        # Trim to the left
        signal = _trim(signal)

        # Trim to the right
        signal.reverse()
        signal = _trim(signal)
        signal.reverse()
        return signal

    def add_silence(self, signal, seconds):
        """
        Add silence to the start and end of 'snd_data' of length 'seconds' (float)
        :param signal:
        :param seconds:
        :return:
        """
        r = array('h', [0 for i in range(int(seconds * self.FREQUENCY))])
        r.extend(signal)
        r.extend([0 for i in range(int(seconds * self.FREQUENCY))])
        return r

    def is_silent(self, signal):
        "Returns 'True' if below the 'silent' threshold"
        return max(signal) < self.SILENCE_THRESHOLD

    def write_to_file(self, signal, sample_width, path):
        #new_signal = pack('<' + ('h' * len(signal)), *signal)
        wf = wave.open(path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(self.FREQUENCY)
        wf.writeframes(signal)
        wf.close()
        self.log.insert(END, 'Endpointed segment is written to output/speech-endpoint.wav')

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()


if __name__ == '__main__':
    sp = SpeechRecognition()
    sp.main()
