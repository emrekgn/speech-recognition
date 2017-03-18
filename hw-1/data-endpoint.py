from sys import byteorder
from array import array
from struct import pack

import pyaudio
import wave
import numpy

ENTROPY_THRESHOLD = 3.8
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
FREQUENCY = 44100
FRAME_SIZE_IN_MS = 0.01
NUM_SAMPLES_PER_FRAME = int(FREQUENCY * FRAME_SIZE_IN_MS)

def normalize(snd_data):
    """
    Average the volume out
    :param snd_data:
    :return:
    """
    MAXIMUM = 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


def trim(snd_data):
    """
    Trim the blank spots at the start and end
    :param snd_data:
    :return:
    """
    THRESHOLD = 500

    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    """
    Add silence to the start and end of 'snd_data' of length 'seconds' (float)
    :param snd_data:
    :param seconds:
    :return:
    """
    r = array('h', [0 for i in range(int(seconds * FREQUENCY))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds * FREQUENCY))])
    return r


def chunks(l, k):
    """
    Yields chunks of size k from a given list.
    """
    for i in range(0, len(l), k):
        yield l[i:i + k]


def shortTermEnergy(frame):
    """
    Calculates the short-term energy of an audio frame. The energy value is
    normalized using the length of the frame to make it independent of said
    quantity.
    """
    return sum([abs(x) ** 2 for x in frame]) / len(frame)


def entropyOfEnergy(frame, numSubFrames):
    """
    Calculates the entropy of energy of an audio frame. For this, the frame is
    partitioned into a number of sub-frames.
    """
    lenSubFrame = int(numpy.floor(len(frame) / numSubFrames))
    shortFrames = list(chunks(frame, lenSubFrame))
    energy = [shortTermEnergy(s) for s in shortFrames]
    totalEnergy = sum(energy)
    energy = [e / totalEnergy for e in energy]

    entropy = 0.0
    for e in energy:
        if e != 0:
            entropy = entropy - e * numpy.log2(e)

    return entropy


def is_silent(chunks):
    """
    Rates an audio sample using its minimum entropy.
    """
    entropy = [entropyOfEnergy(chunk, 20) for chunk in chunks]
    min_entropy = numpy.min(entropy)
    print("Found entropy: {}".format(min_entropy))
    return min_entropy < ENTROPY_THRESHOLD


def begin_recording_with_auto_endpoint():
    """
    Record a word or words from the microphone and
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the
    start and end, and pads with 0.5 seconds of
    blank sound to make sure VLC et al can play
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=FREQUENCY,
                    input=True, output=True,
                    frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        print("Reading data: {}".format(str(snd_data)))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        chunkedData = list(chunks(list(snd_data), NUM_SAMPLES_PER_FRAME))
        silent = is_silent(chunkedData)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 20:
            print("End of speech detected. Stopping recording...")
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Processing data...")
    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def write_sound_file(sample_width, data, path):
    """
    Outputs the resulting data to path
    :param sample_width:
    :param data:
    :param path:
    :return:
    """
    data = pack('<' + ('h' * len(data)), *data)
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(FREQUENCY)
    wf.writeframes(data)
    wf.close()


if __name__ == '__main__':
    input("Press Enter to begin recording...")
    sample_width, data = begin_recording_with_auto_endpoint()
    write_sound_file(sample_width, data, 'endpointed-segment.wav')
    print("Done - result written to endpointed-segment.wav")
