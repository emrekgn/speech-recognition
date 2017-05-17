# -*- coding: utf-8 -*-

import record
import time
import math
import numpy
import scipy.io.wavfile as wav


def endpointed_record():
    """
    record a voice and save it in nonblocking.wav,start to record when you input a number and stop when you stop saying
    """
    rec = record.Recorder(channels=1, rate=44100, endpointed=True)
    with rec.open('output/nonblocking.wav', 'wb') as recfile2:
        input('Press a key to start to recording...')
        recfile2.start_recording()
        time.sleep(100.0)
        recfile2.stop_recording()


def main():
    endpointed_record()


if __name__ == '__main__':
    main()
