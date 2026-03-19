import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 1. Mac 환경 디바이스 설정 (MPS)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# 2. 음성 복제용 Base 모델 로드 (이미 다운로드했으니 이번엔 바로 넘어갑니다!)
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map=device,
    dtype=torch.float32,
)

# 3. 51초짜리 오디오에서 앞 5초만 추출
print("녹음 파일에서 앞부분 5초를 추출합니다...")
audio_data, sr = sf.read("temp_record.wav")
short_audio = audio_data[:sr * 5]
sf.write("temp_record_5s.wav", short_audio, sr)

# 4. 복제 세팅 
text_to_say = "와, 정말 신기하네요! 제 목소리가 단 5초 만에 맥북에서 완벽하게 복제되었습니다."

# 🚨 수정된 부분: 녹음 파일의 첫 5초 동안 '실제로 말씀하신 내용'을 아래에 정확히 타이핑해 주세요!
# (예: "안녕하세요, 저는 지금 음성 복제 테스트를 위해 녹음을...")
ref_text = "여기에 5초 동안 말씀하신 내용을 지우고 적어주세요." 

print("음성을 복제하여 새로운 대사를 생성하는 중입니다...")
wavs, sr_out = model.generate_voice_clone(
    text=text_to_say,
    language="Korean",
    ref_audio="temp_record_5s.wav",
    ref_text=ref_text,  # 이제 에러가 나지 않도록 대본을 함께 넘겨줍니다.
)

print(ref_text)
# 5. 결과 저장
sf.write("output_voice_clone_mac.wav", wavs[0], sr_out)
print("🎉 생성 완료: output_voice_clone_mac.wav 파일이 저장되었습니다!")