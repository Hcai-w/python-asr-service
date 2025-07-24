# python-asr-service/main.py
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from vosk import Model, KaldiRecognizer
import wave
import json
from pydub import AudioSegment
import io
from pydub.silence import split_on_silence

# --- 配置 ---
MODEL_PATH = "vosk-model-small-cn-0.22" # 确保模型文件夹在此路径
VOSK_SAMPLE_RATE = 16000

# --- 初始化 ---
app = FastAPI()

print("正在加载 Vosk 模型...")
try:
    model = Model(MODEL_PATH)
    print("Vosk 模型加载成功!")
except Exception as e:
    print(f"加载 Vosk 模型失败: {e}")
    exit(1)

def recognize_audio(contents: bytes) -> str:
    # 1. 解码为 pydub 的 AudioSegment
    audio = AudioSegment.from_file(io.BytesIO(contents))
    
    # 2. 转换为 16kHz, 16bit, 单声道
    audio = (
        audio.set_frame_rate(VOSK_SAMPLE_RATE)
             .set_channels(1)
             .set_sample_width(2)
    )

    # 3. 静音切割（消除前后/中间停顿影响）
    chunks = split_on_silence(
        audio,
        min_silence_len=300,
        silence_thresh=audio.dBFS - 16,
        keep_silence=150
    )

    if not chunks:
        return ""

    result_text = []
    for chunk in chunks:
        recognizer = KaldiRecognizer(model, VOSK_SAMPLE_RATE)

        # 分片识别（每 200ms）提高语音活动检测可靠性
        step = int(0.2 * VOSK_SAMPLE_RATE) * 2  # 200ms × 16000 × 2bytes
        raw = chunk.raw_data
        for i in range(0, len(raw), step):
            recognizer.AcceptWaveform(raw[i:i + step])
        
        res = json.loads(recognizer.FinalResult())
        text = res.get("text", "")
        if text:
            result_text.append(text)

    return " ".join(result_text)



@app.post("/transcribe")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """
    接收音频文件，进行格式转换，使用 Vosk 识别，并返回文本。
    """
    print(f"收到上传文件: {audio_file.filename}, 类型: {audio_file.content_type}")

    try:
        # 1. 读取上传的音频文件内容
        contents = await audio_file.read()
        transcript = recognize_audio(contents)
        print(f"🗣️ 识别结果: '{transcript}'")
        return {"transcript": transcript}
    except Exception as e:
        print(f"处理音频时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理音频失败: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "Vosk ASR Service is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)