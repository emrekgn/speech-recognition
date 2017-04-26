# -*- coding: utf-8 -*-

import os
import random
import mfcc
import numpy
import DTW
import scipy.io.wavfile as wav


def get_mfcc_feat(filename, winstep=0.01, nfilt=40, numcep=13, preemph=0.95, appendEnergy=True, begin_index=-1,
                  end_index=-1, outputFilename="result.txt"):
    """
    Compute MFCC features from an audio signal.

    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.95.
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.
    :param begin_index: the frame when record begin
    :param end_index: the frame when record end
    :return mfcc_feat: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.
    :return fbank_feat: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.
    :return normalization_feature: the 39-dimensions feature after finite difference and normalization
    """
    (rate, signal) = wav.read(filename)
    if begin_index != -1:
        signal = signal[begin_index * 44100 * 0.02:end_index * 44100 * 0.02]
    if rate == 44100:
        winlen = 0.02
        nfft = 1024
    elif rate == 16000:
        winlen = 0.025
        nfft = 512
    mfcc_feat = mfcc.mfcc(signal=signal, samplerate=rate, winlen=winlen, winstep=winstep, nfft=nfft,
                          nfilt=nfilt, numcep=numcep, preemph=preemph, appendEnergy=appendEnergy)
    numpy.savetxt(outputFilename, mfcc_feat, delimiter=",")
    fbank_feat = mfcc.logfbank(signal=signal, samplerate=rate, winlen=winlen, winstep=winstep, nfft=nfft,
                               nfilt=nfilt, preemph=preemph)
    normalization_feature = mfcc.normalization(mfcc_feat)
    return mfcc_feat, fbank_feat, normalization_feature


def get_templates_and_inputs(digit, number_of_templates):
    """
    split 10 records of a digit to templates and inputs random
    :param digit: the digit you want to split,0-9
    :param number_of_templates: templates' numbers
    :return: a template list and a input list
    """
    templates_index_list = random.sample(xrange(0, 10), number_of_templates)
    inputs_list = []
    templates_list = []
    for i in xrange(0, 10):
        if i in templates_index_list:
            templates_list.append(
                map(list,
                    get_mfcc_feat(filename="./records/" + str(digit) + '_' + str(i) + '.wav', winstep=0.01,
                                  nfilt=40,
                                  numcep=13, preemph=0.95, appendEnergy=False, outputFilename="./output/" + str(digit)
                                                                                              + '_' + str(i) + '.txt')[2]))

        else:
            inputs_list.append(
                map(list,
                    get_mfcc_feat(filename="./records/" + str(digit) + '_' + str(i) + '.wav', winstep=0.01,
                                  nfilt=40,
                                  numcep=13, preemph=0.95, appendEnergy=False, outputFilename="./output/" + str(digit)
                                                                                              + '_' + str(i) + '.txt')[2]))
    return templates_list, inputs_list


def training_model(number_of_templates, using_kmeans):
    """
    training the model using DTW to recognize the records
    :param number_of_templates: int, using number_of_templates to train the model
    :param using_kmeans: 1 if using k-means to templates to generate some states
    :return: DTW_obj:a DTW object,the trained model
    :return: inputs:list,every element of it is a template using for test
    """
    templates = []
    inputs = []
    for digit in xrange(0, 10):
        temp = get_templates_and_inputs(digit, number_of_templates)
        templates.extend(temp[0])
        inputs.extend(temp[1])
    DTW_obj = DTW.DTW(templates)
    return DTW_obj, inputs


def test(time_synchronous, pruning, number_of_templates, using_kmeans, begin_index, end_index, DTW_obj, inputs=[]):
    """
    :param time_synchronous: 0: using recorded data as input
                             1: using time synchronous record as input
    :param begin_index: the frame when record begin
    :param end_index: the frame when record end
    :param pruning: pruning if equals to 1
    :param using_kmeans: 1 if using k-means to templates to generate some states
    :param: DTW_obj:a DTW object,the trained model
    :param: inputs:list,every element of it is a template using for test
    """
    if not time_synchronous:
        # when pruning equals to 0,use numbers of templates 1-5 to compare once,otherwise use 5 templates to compare until accuracy is big enough
        if pruning == 0 or number_of_templates == 5:
            accuracy = 0.0
            total_accuracy = 1.0  # product of every loop's accuracy
            while accuracy < 0.7:
                wrong = 0
                for i in xrange(0, len(inputs)):
                    right_digit = i / (10 - number_of_templates)
                    distance, template_index, path = DTW_obj.DTW(inputs[i], strategy=pruning,
                                                                 accuracy=total_accuracy)
                    predict_digit = template_index if using_kmeans else template_index / number_of_templates
                    if right_digit != predict_digit:
                        wrong += 1
                accuracy = 1 - float(wrong) / len(inputs)
                total_accuracy *= accuracy
                log('\nwhen using ' + str(number_of_templates) + ' templates the accuracy is' + str(accuracy) + '\n')
                if pruning == 0:
                    break
    else:
        if number_of_templates == 5:
            input = map(list, get_mfcc_feat(filename='recording.wav', winstep=0.01, \
                                            nfilt=40, numcep=13, preemph=0.95, appendEnergy=False, \
                                            begin_index=begin_index, end_index=end_index)[2])
            distance, template_index, path = DTW_obj.DTW(input, 0)
            predict_digit = template_index if using_kmeans else  template_index / number_of_templates
            log('\nwhen using ' + str(number_of_templates) + ' predict you spoke' + str(predict_digit) + '\n')


def test_DTW(time_synchronous=0, begin_index=0, end_index=0, pruning=0, using_kmeans=0):
    """
    test all records in inputs list using different number of templates(1-5)
    :param time_synchronous: 0: using recorded data as input
                             1: using time synchronous record as input
    :param begin_index: the frame when record begin
    :param end_index: the frame when record end
    :param pruning: pruning if equals to 1
    :param using_kmeans: 1 if using k-means to templates to generate some states
    """
    for number_of_templates in xrange(1, 6):
        if number_of_templates == 5 or (not using_kmeans):
            DTW_obj, inputs = training_model(number_of_templates, using_kmeans)
            test(time_synchronous, pruning, number_of_templates, using_kmeans, begin_index, end_index, DTW_obj, inputs)


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
    log('\nthis is the result using DTW without pruning\n')
    test_DTW(pruning=0, using_kmeans=0)
    log('\nthis is the result using DTW with pruning\n')
    test_DTW(pruning=1, using_kmeans=0)


if __name__ == '__main__':
    main()
