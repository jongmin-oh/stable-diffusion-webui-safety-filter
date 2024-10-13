echo "Downloading..."

# Checkpoints
if [ ! -f "models/Stable-diffusion/koreanBoyMerge_v2.safetensors" ]; then
    curl -L -o "models/Stable-diffusion/koreanBoyMerge_v2.safetensors" "https://huggingface.co/j5ng/sd-image-generate-models/resolve/main/checkpoints/koreanBoyMerge_v2.safetensors"
else
    echo "models/Stable-diffusion/koreanBoyMerge_v2.safetensors 파일이 이미 존재합니다. 건너뜁니다."
fi

if [ ! -f "models/Stable-diffusion/lemixRealisticAsians_v3_fp16.safetensors" ]; then
    curl -L -o "models/Stable-diffusion/lemixRealisticAsians_v3_fp16.safetensors" "https://huggingface.co/j5ng/sd-image-generate-models/resolve/main/checkpoints/lemixRealisticAsians_v3_fp16.safetensors"
else
    echo "models/Stable-diffusion/lemixRealisticAsians_v3_fp16.safetensors 파일이 이미 존재합니다. 건너뜁니다."
fi

if [ ! -f "models/Stable-diffusion/blueboys2D_Merge_v30.safetensors" ]; then
    curl -L -o "models/Stable-diffusion/blueboys2D_Merge_v30.safetensors" "https://huggingface.co/j5ng/sd-image-generate-models/resolve/main/checkpoints/blueboys2D_Merge_v30.safetensors"
else
    echo "models/Stable-diffusion/blueboys2D_Merge_v30.safetensors 파일이 이미 존재합니다. 건너뜁니다."
fi

# Lora
mkdir -p "models/Lora"
if [ ! -f "models/Lora/GoodHands-beta2.safetensors" ]; then
    curl -L -o "models/Lora/GoodHands-beta2.safetensors" "https://huggingface.co/j5ng/sd-image-generate-models/resolve/main/Lora/GoodHands-beta2.safetensors"
else
    echo "models/Lora/GoodHands-beta2.safetensors 파일이 이미 존재합니다. 건너뜁니다."
fi

if [ ! -f "models/Lora/koreanDollLikeness_v15.safetensors" ]; then
    curl -L -o "models/Lora/koreanDollLikeness_v15.safetensors" "https://huggingface.co/j5ng/sd-image-generate-models/resolve/main/Lora/koreanDollLikeness_v15.safetensors"
else
    echo "models/Lora/koreanDollLikeness_v15.safetensors 파일이 이미 존재합니다. 건너뜁니다."
fi

# VAE
if [ ! -f "models/VAE/vae-ft-mse-840000-ema-pruned.safetensors" ]; then
    curl -L -o "models/VAE/vae-ft-mse-840000-ema-pruned.safetensors" "https://huggingface.co/j5ng/sd-image-generate-models/resolve/main/VAE/vae-ft-mse-840000-ema-pruned.safetensors"
else
    echo "models/VAE/vae-ft-mse-840000-ema-pruned.safetensors 파일이 이미 존재합니다. 건너뜁니다."
fi

if [ ! -f "models/VAE/klF8Anime2VAE_klF8Anime2VAE.safetensors" ]; then
    curl -L -o "models/VAE/klF8Anime2VAE_klF8Anime2VAE.safetensors" "https://huggingface.co/j5ng/sd-image-generate-models/resolve/main/VAE/klF8Anime2VAE_klF8Anime2VAE.safetensors"
else
    echo "models/VAE/klF8Anime2VAE_klF8Anime2VAE.safetensors 파일이 이미 존재합니다. 건너뜁니다."
fi

# Embeddings


echo "Done!"
