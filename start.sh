#!/bin/bash

# --- 启动 FastAPI ASR 服务 ---
echo "--- 启动 FastAPI ASR 服务 ---"
# Render 或 Hugging Face Spaces 会通过 $PORT 环境变量告诉我们应该使用哪个端口
uvicorn main:app --host 0.0.0.0 --port $PORT