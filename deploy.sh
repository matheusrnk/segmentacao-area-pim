#!/bin/bash

if [ -d ".venv" ]
then
    . .venv/bin/activate
    pip3 install -r requirements.txt
    echo "Forneça o caminho para imagem desejada: "
    read filepath
    python3 main.py $filepath
else
    python3 -m venv .venv
    . .venv/bin/activate
    python3 -m pip3 install --upgrade pip
    pip3 install -r requirements.txt
    echo "Forneça o caminho para imagem desejada: "
    read $filepath
    python3 main.py $filepath
fi