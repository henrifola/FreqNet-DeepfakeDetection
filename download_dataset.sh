#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# === Setup ===
BASE_DIR="dataset"
TEST_DIR="$BASE_DIR/test"
TRAIN_DIR="$BASE_DIR/train"
VAL_DIR="$BASE_DIR/val"
FORENSYNTHS_DIR="$TEST_DIR/ForenSynths"

mkdir -p "$BASE_DIR"
cd "$BASE_DIR"
echo "Current directory: $(pwd)"
echo "Starting dataset setup..."

# ---------- CNNDetection TEST SET ----------
if [ ! -d "$FORENSYNTHS_DIR" ]; then
    echo "[INFO] Downloading CNNDetection TEST set..."
    mkdir -p "$FORENSYNTHS_DIR"
    if [ ! -f "$FORENSYNTHS_DIR/CNN_synth_testset.zip" ]; then
        wget -q https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/CNN_synth_testset.zip -O "$FORENSYNTHS_DIR/CNN_synth_testset.zip"
    fi
    echo "[INFO] Unzipping CNNDetection TEST set..."
    unzip -o "$FORENSYNTHS_DIR/CNN_synth_testset.zip" -d "$FORENSYNTHS_DIR/"
    rm "$FORENSYNTHS_DIR/CNN_synth_testset.zip"
    echo "✅ CNNDetection TEST set ready."
else
    echo "[SKIP] CNNDetection TEST set already exists."
fi


# ---------- CNNDetection TRAIN SET ----------
if [ ! -d "$TRAIN_DIR" ]; then
    echo "[INFO] Downloading CNNDetection TRAIN set..."
    mkdir -p "$TRAIN_DIR"
    for i in {001..007}; do
        file="progan_train.7z.$i"
        if [ ! -f "$file" ]; then
            echo "[INFO] Downloading $file ..."
            wget -q https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/$file
        else
            echo "[SKIP] $file already exists."
        fi
    done

    echo "[INFO] Extracting CNNDetection TRAIN set..."
    7z x progan_train.7z.001

    if [ -f "progan_train.zip" ]; then
        mv progan_train.zip "$TRAIN_DIR/"
        unzip -o "$TRAIN_DIR/progan_train.zip" -d "$TRAIN_DIR/"
        rm "$TRAIN_DIR/progan_train.zip"
    fi

    rm progan_train.7z.*
    echo "✅ CNNDetection TRAIN set ready."
else
    echo "[SKIP] CNNDetection TRAIN set already exists."
fi


# ---------- CNNDetection VAL SET ----------
if [ ! -d "$VAL_DIR" ]; then
    echo "[INFO] Downloading CNNDetection VAL set..."
    mkdir -p "$VAL_DIR"
    if [ ! -f "$VAL_DIR/progan_val.zip" ]; then
        wget -q https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip -O "$VAL_DIR/progan_val.zip"
    fi
    echo "[INFO] Unzipping CNNDetection VAL set..."
    unzip -o "$VAL_DIR/progan_val.zip" -d "$VAL_DIR/"
    rm "$VAL_DIR/progan_val.zip"
    echo "✅ CNNDetection VAL set ready."
else
    echo "[SKIP] CNNDetection VAL set already exists."
fi


# ---------- GANGen-Detection ---------- #
if [ ! -d "../GANGen-Detection" ]; then
    echo "[INFO] Downloading GANGen-Detection dataset..."
    gdown https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj -O ../GANGen-Detection --folder

    cd ../GANGen-Detection
    for file in *.tar.gz *.tgz *.tar; do
        [ -e "$file" ] || continue
        echo "[INFO] Extracting $file..."
        tar -zxvf "$file"
        rm "$file"
    done
    cd "$pwd/dataset"
else
    echo "[SKIP] GANGen-Detection already exists."
fi

# ---------- Done ----------
echo "Dataset structure after setup:"
tree "$BASE_DIR"
echo "[DONE] All datasets are ready!"
