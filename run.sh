#!/bin/bash

# set up bash to handle errors more aggressively - a "strict mode" of sorts
set -e # give an error if any command finishes with a non-zero exit code
set -u # give an error if we reference unset variables
set -o pipefail # for a pipeline, if any of the commands fail with a non-zero exit code, fail the entire pipeline with that exit code

echo "Installing dependencies..."
apt-get update
apt-get install -y build-essential python3 python3-setuptools python3-pip python3-pyaudio python3-tk
pip3 install --upgrade pip
pip3 install python_speech_features scipy matplotlib numpy
echo "Installed dependencies."

echo "Starting speech-recognition..."
python3 speech_recognition.py
echo "Finished speech-recognition."
