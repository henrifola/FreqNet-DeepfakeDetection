#!/bin/bash

set -e

pwd=$(cd $(dirname $0); pwd)
echo "Current directory: $pwd"

mkdir -p dataset
cd dataset

# ---------- CNNDetection TEST SET ----------
if [ ! -d "ForenSynths/test" ]; then
    echo "[INFO] Downloading CNNDetection TEST set..."
    if [ ! -f "CNN_synth_testset.zip" ]; then
        wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip
    fi
    unzip -o CNN_synth_testset.zip -d ForenSynths
    rm CNN_synth_testset.zip
else
    echo "[SKIP] CNNDetection TEST set already exists."
fi

# ---------- CNNDetection VAL SET ----------
if [ ! -d "ForenSynths/val" ]; then
    echo "[INFO] Downloading CNNDetection VAL set..."
    if [ ! -f "progan_val.zip" ]; then
        wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip
    fi
    unzip -o progan_val.zip -d ForenSynths
    rm progan_val.zip
else
    echo "[SKIP] CNNDetection VAL set already exists."
fi

# ---------- CNNDetection TRAIN SET ----------
if [ ! -d "ForenSynths/train" ]; then
    echo "[INFO] Downloading CNNDetection TRAIN set..."
    for i in {001..007}; do
        file="progan_train.7z.$i"
        if [ ! -f "$file" ]; then
            echo "[INFO] Downloading $file ..."
            wget https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/$file
        else
            echo "[SKIP] $file already exists."
        fi
    done

    echo "[INFO] Extracting .7z files..."
    7z x -y progan_train.7z.001

    if [ -f "progan_train.zip" ]; then
        unzip -o progan_train.zip -d ForenSynths
        rm progan_train.zip
    fi

    rm progan_train.7z.*
else
    echo "[SKIP] CNNDetection TRAIN set already exists."
fi

# ---------- GANGen-Detection ----------
if [ ! -d "GANGen-Detection" ]; then
    echo "[INFO] Downloading GANGen-Detection dataset..."
    gdown https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj -O ./GANGen-Detection --folder

    cd GANGen-Detection
    for file in *.tar.gz *.tgz *.tar; do
        [ -e "$file" ] || continue
        echo "[INFO] Extracting $file"
        tar -zxvf "$file" && rm "$file"
    done
    cd ..
else
    echo "[SKIP] GANGen-Detection already exists."
fi

echo "[DONE] All datasets are ready!"
