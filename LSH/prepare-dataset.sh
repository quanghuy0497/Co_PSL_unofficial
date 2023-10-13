#! /bin/bash

mkdir -p dataset
cd dataset
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd ..
./convert.py
