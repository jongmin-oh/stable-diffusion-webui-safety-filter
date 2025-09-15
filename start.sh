nohup ./webui.sh --listen --api --xformers --opt-sdp-attention --disable-nan-check --port 7860 & 
sleep 5  # 5초 대기

nohup ./webui.sh --listen --api --xformers --opt-sdp-attention --disable-nan-check --port 7861 &

sleep 5  # 5초 대기
nohup ./webui.sh --listen --api --xformers --opt-sdp-attention --disable-nan-check --port 7862 &

sleep 5  # 5초 대기
nohup ./webui.sh --listen --api --xformers --opt-sdp-attention --disable-nan-check --port 7863 &

sleep 20
nvidia-smi