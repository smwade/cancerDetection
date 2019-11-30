#! /bin/sh
#
# Prepare environment

export $(egrep -v '^#' .env | xargs)
pip install -e ./cancer

