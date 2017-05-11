#!/usr/bin/env bash

set -e
set -u
set -o pipefail

apt-get update
apt-get install -y --force-yes build-essential python python-dev python-all-dev virtualenv python-pip gcc libncurses5-dev libxml2-dev libxslt1-dev swig git libpulse-dev

