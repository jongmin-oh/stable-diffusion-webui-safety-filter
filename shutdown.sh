#!/bin/bash

echo "GPU 프로세스 종료 시작..."

# nvidia-smi에서 실행 중인 모든 프로세스 PID 추출
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)

if [ -z "$gpu_pids" ]; then
    echo "실행 중인 GPU 프로세스가 없습니다."
    exit 0
fi

# 각 PID에 대해 종료 처리
for pid in $gpu_pids; do
    echo "PID $pid 프로세스 정보:"
    ps -f -p $pid
    
    echo "PID $pid 종료 시도 중..."
    kill $pid
    sleep 2  # 종료 대기
    
    # 여전히 실행 중이라면 강제 종료
    if ps -p $pid > /dev/null; then
        echo "PID $pid 강제 종료 중..."
        kill -9 $pid
    fi
    
    echo "PID $pid 종료 완료"
    echo "------------------------"
done

echo "모든 GPU 프로세스 종료 완료"

nvidia-smi