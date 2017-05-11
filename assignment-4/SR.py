# -*- coding: utf-8 -*-

import speech_recognition as sr

KEYWORDS = [("one", 1.15), ("two", 1.15), ("three", 0.75), ("four", 1.15), ("five", 1.15), ("six", 1.15), ("seven", 0.75),
            ("eight", 1.15), ("nine", 1.15), ("zero", 1.15)]


def main():
    # obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Running...")
        r.adjust_for_ambient_noise(source)
        print("Say something!")
        audio = r.listen(source)

    # recognize speech using Sphinx
    print("Trying to recognize...")
    try:
        #print("You said: \"" + r.recognize_sphinx(audio, keyword_entries=KEYWORDS, show_all=False) + "\"")
        print("You said: \"" + r.recognize_sphinx(audio, show_all=False) + "\"")
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError as e:
        print("Error; {0}".format(e))

if __name__ == '__main__':
    main()
