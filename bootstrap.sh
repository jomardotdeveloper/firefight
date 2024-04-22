#!/bin/bash

sudo rfcomm bind /dev/rfcomm0 98:DA:50:01:CE:08
../venv/bin/python server.py
