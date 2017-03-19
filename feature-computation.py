#!/usr/bin/env python

from __future__ import division
from sys import byteorder
from array import array
from struct import pack
import tkinter
from python_speech_features import mfcc
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy
import pyaudio
import wave

# Tkinter GUI
root = tkinter.Tk()

# Common parameters
CHUNK_SIZE = 1024 # num of frames to read at a time
FORMAT = pyaudio.paInt16
FREQUENCY = 44100
FRAME_SIZE_IN_MS = 0.01
NUM_SAMPLES_PER_FRAME = int(FREQUENCY * FRAME_SIZE_IN_MS)
ENERGY_THRESHOLD = 50
SILENCE_THRESHOLD = 500
MAX_SILENCE_COUNT = 10

OUTPUT_WAV = 'output/speech-endpoint.wav'
OUTPUT_ZERO_CROSSING_RATE = 'output/zero-crossing-rate.png'
OUTPUT_ENERGY = 'output/short-time-energy.png'
OUTPUT_SPECTROGRAM = 'output/spectrogram.png'
OUTPUT_RAW_SIGNAL = 'output/raw-signal.png'


def normalize(signal):
    """
    Average the volume out
    :param signal:
    :return:
    """
    #signal = signal / max(abs(signal))  # scale signal
    MAXIMUM = 16384
    times = float(MAXIMUM) / max(abs(i) for i in signal)

    new_signal = array('h')
    for i in signal:
        new_signal.append(int(i * times))
    return new_signal


def trim(signal):
    """
    Trim the blank spots at the start and end
    :param signal:
    :return:
    """
    def _trim(snd_data):
        snd_started = False
        new_signal = array('h')

        for i in snd_data:
            if not snd_started and abs(i) > SILENCE_THRESHOLD:
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


def add_silence(signal, seconds):
    """
    Add silence to the start and end of 'snd_data' of length 'seconds' (float)
    :param signal:
    :param seconds:
    :return:
    """
    r = array('h', [0 for i in range(int(seconds * FREQUENCY))])
    r.extend(signal)
    r.extend([0 for i in range(int(seconds * FREQUENCY))])
    return r


def is_silent(signal):
    "Returns 'True' if below the 'silent' threshold"
    return max(signal) < SILENCE_THRESHOLD


def write_to_file(signal, sample_width, path):
    new_signal = pack('<' + ('h'*len(signal)), *signal)
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(FREQUENCY)
    wf.writeframes(new_signal)
    wf.close()


if __name__ == '__main__':

    input("Press Enter to begin recording...")

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=FREQUENCY,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)
    sample_width = p.get_sample_size(FORMAT)

    num_silence = 0
    snd_started = False

    signal = array('h')
    while 1:
        # little endian, signed short
        chunk = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            chunk.byteswap()
        signal.extend(chunk)

        silent = is_silent(chunk)

        if silent and snd_started:
            num_silence += 1
        elif not silent and not snd_started:
            snd_started = True
        if snd_started and num_silence > MAX_SILENCE_COUNT:
            print("End of speech detected. Stopping recording...")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    signal = normalize(signal)
    signal = trim(signal)
    signal = add_silence(signal, 0.5)
    write_to_file(signal, sample_width, OUTPUT_WAV)
    signal = numpy.fromiter(signal, dtype=numpy.int16)

    signal = signal / max(abs(signal)) # scale for plotting and calculations
    assert min(signal) >= -1 and max(signal) <= 1

    # Prints some stats
    print('frequency ==> {} Hz'.format(FREQUENCY)) # sampling rate
    print('length of signal  ==> {} samples'.format(len(signal)))
    print('signal  ==> {}'.format(signal))

    sampsPerMilli = int(FREQUENCY / 1000)
    millisPerFrame = 20
    sampsPerFrame = sampsPerMilli * millisPerFrame
    nFrames = int(len(signal) / sampsPerFrame)  # number of non-overlapping _full_ frames

    print('samples/millisecond  ==> {}'.format(sampsPerMilli))
    print('samples/[%dms]frame  ==> % {} {}'.format(millisPerFrame, sampsPerFrame))
    print('number of frames     ==> {}'.format(nFrames))

    # Raw signal
    plt.figure()
    plt.plot(signal)
    plt.title('Raw Signal')
    plt.xlabel('Sample')
    plt.autoscale(tight='both')
    plt.savefig(OUTPUT_RAW_SIGNAL)

    # Spectrogram
    N = len(signal)  # total number of signals
    curPos = 0
    Win = round(FREQUENCY * 0.040)
    Step = round(FREQUENCY * 0.040)
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
    FreqTicksLabels = [str(FREQUENCY / 2 - int((f * FREQUENCY) / (2 * nfft))) for f in FreqTicks]
    ax.set_yticks(FreqTicks)
    ax.set_yticklabels(FreqTicksLabels)
    TStep = round(countFrames / 3)
    TimeTicks = range(0, countFrames, TStep)
    TimeTicksLabels = ['%.2f' % (float(t * Step) / FREQUENCY) for t in TimeTicks]
    ax.set_xticks(TimeTicks)
    ax.set_xticklabels(TimeTicksLabels)
    ax.set_xlabel('time (secs)')
    ax.set_ylabel('freq (Hz)')
    imgplot.set_cmap('jet')
    plt.colorbar()
    plt.savefig(OUTPUT_SPECTROGRAM)

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
    plt.title('Short-Time Energy')
    plt.ylabel('ENERGY')
    plt.xlabel('FRAME')
    plt.autoscale(tight='both')
    plt.savefig(OUTPUT_ENERGY)

    # Zero-crossing rate
    DC = numpy.mean(signal)
    newSignal = signal - DC  # create a new signal, preserving old
    print('DC               ==> {}'.format(DC))
    print('mean(newSignal)  ==> {}'.format(numpy.mean(newSignal)))
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
    plt.title('Short-Time Zero Crossing Counts')
    plt.ylabel('ZCC')
    plt.xlabel('FRAME')
    plt.autoscale(tight='both')
    plt.savefig(OUTPUT_ZERO_CROSSING_RATE)
    plt.show()

    # Extract features
    mfcc_feat_40 = mfcc(signal, FREQUENCY, nfilt=40, lowfreq=50)
    numpy.savetxt('output/mfcc-nf40.txt', mfcc_feat_40)
    mfcc_feat_30 = mfcc(signal, FREQUENCY, nfilt=30, lowfreq=50)
    numpy.savetxt('output/mfcc-nf30.txt', mfcc_feat_30)
    mfcc_feat_25 = mfcc(signal, FREQUENCY, nfilt=25, lowfreq=50)
    numpy.savetxt('output/mfcc-nf25.txt', mfcc_feat_25)

    print("Done - results written to output directory")
    root.mainloop()
