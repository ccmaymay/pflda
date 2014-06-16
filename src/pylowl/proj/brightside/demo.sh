#!/bin/bash

pushd ../../../../ # repo root
python setup.py build --with-proj-brightside
popd
