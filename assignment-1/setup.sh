#!/usr/bin/env bash

set -e
set -u
set -o pipefail

apt-get update
apt-get install -y --force-yes build-essential python python-dev gcc libncurses5-dev libxml2-dev libxslt1-dev python-pyaudio portaudio19-dev

