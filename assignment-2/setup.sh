#!/usr/bin/env bash

set -e
set -u
set -o pipefail

apt-get update
apt-get install -y --force-yes build-essential python3 python3-dev python3-tk gcc libncurses5-dev libxml2-dev libxslt1-dev python3-pyaudio portaudio19-dev

