#!/usr/bin/env python

from __future__ import division
from sys import byteorder
from array import array
import os
from python_speech_features import mfcc
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy
import pyaudio
import wave

FREQUENCY = 44100
INPUT_PATH = 'record/record.wav'
OUTPUT_ZERO_CROSSING_RATE = 'output/zero-crossing-rate.png'
OUTPUT_ENERGY = 'output/short-time-energy.png'
OUTPUT_SPECTROGRAM = 'output/spectrogram.png'
OUTPUT_RAW_SIGNAL = 'output/raw-signal.png'
OUTPUT_MFCC_TXT = 'output/mfcc-features.txt'
OUTPUT_MFCC_FIG = 'output/mfcc-features.png'


def analyze(signal):
    log('Analyzing audio signal...\n')
    signal = signal / max(abs(signal))  # scale for plotting and calculations
    assert min(signal) >= -1 and max(signal) <= 1

    # Prints some stats
    log('Frequency ==> {} Hz\n'.format(FREQUENCY))  # sampling rate
    log('Length of signal  ==> {} samples\n'.format(len(signal)))
    log('Signal  ==> {}\n'.format(signal))

    sampsPerMilli = int(FREQUENCY / 1000)
    millisPerFrame = 20
    sampsPerFrame = sampsPerMilli * millisPerFrame
    nFrames = int(len(signal) / sampsPerFrame)  # number of non-overlapping _full_ frames

    log('Samples/millisecond  ==> {}\n'.format(sampsPerMilli))
    log('Samples/[%dms]frame  ==> % {} {}\n'.format(millisPerFrame, sampsPerFrame))
    log('Number of frames     ==> {}\n'.format(nFrames))

    # Raw signal
    plt.figure()
    plt.plot(signal)
    plt.title('Raw Signal')
    plt.xlabel('Sample')
    plt.autoscale(tight='both')
    plt.savefig(OUTPUT_RAW_SIGNAL)

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
    plt.savefig(OUTPUT_ENERGY)

    # Zero-crossing rate
    DC = numpy.mean(signal)
    newSignal = signal - DC  # create a new signal, preserving old
    log('DC               ==> {}\n'.format(DC))
    log('mean(newSignal)  ==> {}\n'.format(numpy.mean(newSignal)))
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
    plt.savefig(OUTPUT_ZERO_CROSSING_RATE)

    # Extract features
    mfcc_features = mfcc(signal, FREQUENCY, nfilt=40, lowfreq=50)
    numpy.savetxt(OUTPUT_MFCC_TXT, mfcc_features)
    log('MFCC:\nNumber of windows = {}\n'.format(mfcc_features.shape[0]))
    log('Length of each feature = {}\n'.format(mfcc_features.shape[1]))

    # plt.figure()
    # Transform the matrix so that the time domain is horizontal
    mfcc_features = mfcc_features.T
    plt.matshow(mfcc_features)
    plt.title('MFCC')
    plt.savefig(OUTPUT_MFCC_FIG)
    #plt.show()

    log("Done - results written to output directory.\n")


def log(message):
    """
    Appends provided message into ./output/log.txt
    :param message: 
    :return: 
    """
    if not os.path.exists('./output'):
        os.makedirs('./output')
    with open('./output/log.txt', 'a+') as f:
        f.write(message)


def main():
    global FREQUENCY
    log('Reading endpointed audio file...\n')
    FREQUENCY, signal = wavfile.read(INPUT_PATH)
    analyze(signal)


if __name__ == '__main__':
    main()
