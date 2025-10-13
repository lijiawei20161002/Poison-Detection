MODEL=meta-llama/Llama-2-7b
DEST=$HOME/models/$(basename "$MODEL")

# pull every file, including real LFS blobs
huggingface-cli download "$MODEL" \
    --local-dir "$DEST" \
    --local-dir-use-symlinks False