# -*- coding: utf-8 -*-

import speech_recognition as sr
import os

KEYWORDS = [("one", 1.0), ("two", 1.0), ("three", 1.0), ("four", 1.0), ("five", 1.0), ("six", 1.0), ("seven", 1.0),
            ("eight", 1.0), ("nine", 1.0), ("zero", 1.0)]


def main():
    r = sr.Recognizer()
    for f in os.listdir("./records"):
        if f.endswith(".wav"):
            audio_file = os.path.join("./records", f)
            print("Reading file {0}...".format(audio_file))
            with sr.AudioFile(audio_file) as source:
                audio = r.record(source)  # read the entire audio file
                print("Trying to recognize...")
                try:
                    print("Record says: \"" + r.recognize_sphinx(audio, show_all=False) + "\"\n")
                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print("Error; {0}".format(e))

    # obtain audio from the microphone
    with sr.Microphone() as source:
        print("Running...")
        r.adjust_for_ambient_noise(source)
        print("Say something!")
        audio = r.listen(source)
    print("Trying to recognize...")
    try:
        print("You said: \"" + r.recognize_sphinx(audio, show_all=False) + "\"")
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Error; {0}".format(e))

if __name__ == '__main__':
    main()
