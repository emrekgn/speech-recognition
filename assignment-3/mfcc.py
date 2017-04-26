# -*- coding: utf-8 -*-
# based on this project:https://github.com/jameslyons/python_speech_features
# calculate filterbank features. Provides e.g. fbank and mfcc features for use in ASR applications
import numpy
import math
import sigproc
from scipy.fftpack import dct


def mfcc(signal, samplerate=16000, winlen=0.025, winstep=0.01, numcep=13,
         nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.95, ceplifter=22, appendEnergy=True):
    """
    Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)    
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)    
    :param numcep: the number of cepstrum to return, default 13    
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.95. 
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22. 
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    """
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    if appendEnergy: feat[:, 0] = numpy.log(energy)  # replace first cepstral coefficient with log of frame energy
    return feat


def fbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
          nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.95):
    """
    Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)    
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)    
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.95. 
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The
        second return value is the energy in each frame (total energy, unwindowed)
    """
    highfreq = highfreq or samplerate / 2
    signal = sigproc.preemphasis(signal, preemph)
    # print type(signal[0])
    frames = sigproc.framesig(signal, winlen * samplerate, winstep * samplerate, winfunc=hamming_window)
    powspec = sigproc.powspec(frames, nfft)
    # numpy.savetxt("result.txt", powspec, delimiter=",")
    energy = numpy.sum(powspec, 1)  # this stores the total energy in each frame
    energy = numpy.where(energy == 0, numpy.finfo(float).eps,
                         energy)  # if energy is zero, we get problems with log, use numpy.finfo(float).eps to replace 0

    filterbanks = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    # print powspec.shape, filterbanks.shape
    feat = numpy.dot(powspec, filterbanks.T)  # compute the filterbank energies
    feat = numpy.where(feat == 0, numpy.finfo(float).eps, feat)  # if feat is zero, we get problems with logs
    # print feat.shape
    return feat, energy


def logfbank(signal, samplerate=16000, winlen=0.025, winstep=0.01,
             nfilt=26, nfft=512, lowfreq=0, highfreq=None, preemph=0.95):
    """
    Compute log Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)    
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)    
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.95. 
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. 
    """
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph)
    return numpy.log(feat)


def hz2mel(hz):
    """
    Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1 + hz / 700.0)


def mel2hz(mel):
    """
    Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700 * (10 ** (mel / 2595.0) - 1)


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """
    Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    # We do n filterbanks, for which we need n+2 points.
    melpoints = numpy.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert from Hz to fft bin number
    bin = numpy.floor((nfft + 1) * mel2hz(melpoints) / samplerate)
    # print bin
    fbank = numpy.zeros([nfilt, nfft / 2 + 1])
    for j in xrange(0, nfilt):
        for i in xrange(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / pow((bin[j + 1] - bin[j]), 2)
        for i in xrange(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / pow((bin[j + 2] - bin[j + 1]), 2)
            # print max(fbank[j])
    return fbank


def hamming_window(x):
    """
    Add hamming window to frames

    :param x: the length of the window
    :return: a numpy array of size 1 * x
    """
    window = []
    for i in range(x):
        window.append(0.54 - 0.46 * math.cos(2.0 * math.pi * i / x))
    # print 'window\n', window
    return numpy.array(window)


def normalization(origin_feature):
    """
    get the 39-dimensions feature after finite difference and normalization
    :param array,origin_feature:origin mfcc feature
    :return:39-dimensions feature
    """
    number_of_frames = origin_feature.shape[0]
    mean = numpy.zeros(13)
    for i in range(0, number_of_frames):
        numpy.add(mean, origin_feature[i], mean)
    numpy.divide(mean, number_of_frames, mean)
    for i in xrange(0, number_of_frames):
        numpy.subtract(origin_feature[i], mean, origin_feature[i])
    normalization_feature = []
    temp_normalization_feature = []
    for i in xrange(number_of_frames):
        if i == 0:
            temp = numpy.append(origin_feature[i], origin_feature[i + 1])
        elif i == origin_feature.shape[0] - 1:
            temp = numpy.append(origin_feature[i], -origin_feature[i - 1])
        else:
            temp = numpy.append(origin_feature[i], numpy.subtract \
                (origin_feature[i + 1], origin_feature[i - 1]))
        temp_normalization_feature.append(temp)
    for i in xrange(number_of_frames):
        if i == 0:
            temp = numpy.append(temp_normalization_feature[i], temp_normalization_feature[i + 1][13:26])
        elif i == len(temp_normalization_feature) - 1:
            temp = numpy.append(temp_normalization_feature[i], -temp_normalization_feature[i - 1][13:26])
        else:
            temp = numpy.append(temp_normalization_feature[i], numpy.subtract \
                (temp_normalization_feature[i + 1][13:26], temp_normalization_feature[i - 1][13:26]))
        normalization_feature.append(temp)
    normalization_feature = numpy.array(normalization_feature)
    # print normalization_feature[0]
    mean = numpy.zeros(39)
    for i in range(0, number_of_frames):
        numpy.add(mean, normalization_feature[i], mean)
    numpy.divide(mean, number_of_frames, mean)
    # print mean
    for i in range(0, number_of_frames):
        numpy.subtract(normalization_feature[i], mean, normalization_feature[i])
    # print normalization_feature[0]
    return normalization_feature
