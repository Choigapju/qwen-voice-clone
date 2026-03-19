import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 1. Mac 환경에 맞춘 디바이스 설정 (MPS 사용)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"현재 사용 중인 디바이스: {device}")

# 2. 모델 로드 (Mac 호환성을 위해 float32 사용, flash_attention 제외)
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map=device,
    dtype=torch.float32,
)

# 3. 음성 생성 (Voice Design)
text = "안녕하세요? 제 목소리는 맥북에서 텍스트 설명만으로 만들어졌습니다. 정말 신기하죠?"
instruct = "차분하고 신뢰감 있는 중저음의 남성 뉴스 앵커 목소리, 전문적인 톤."

print("음성을 생성하는 중입니다. 잠시만 기다려주세요...")
wavs, sr = model.generate_voice_design(
    text=text,
    language="Korean",
    instruct=instruct
)

# 4. 결과 저장
sf.write("output_voice_design_mac.wav", wavs[0], sr)
print("생성 완료: output_voice_design_mac.wav 파일이 현재 폴더에 저장되었습니다.")