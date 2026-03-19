import gradio as gr
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import os
from transformers import pipeline

# 디바이스 설정 (Mac CPU/GPU 호환)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading Qwen3-TTS model on {device}...")

# 1. 앱 시작 시 한 번만 모델을 로드하여 대기 시간을 줄입니다.
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map=device,
    dtype=torch.float32,
)
print("Qwen3-TTS Model loaded successfully!")

# Whisper 모델 로드 (STT용)
print(f"Loading Whisper STT model on {device}...")
stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)
print("Whisper Model loaded successfully! Ready to serve.")

def clone_voice(audio_path, ref_text, target_text):
    if not audio_path:
        return None, "오디오 파일을 업로드하거나 녹음해주세요."
    if not target_text:
        return None, "생성할 대사를 입력해주세요."
    
    try:
        # 2. 업로드된 오디오 읽기
        audio_data, sr = sf.read(audio_path)
        
        # 3. 앞부분 5초만 추출 (모델 내부에서 5초 추출하여 복제 세팅)
        max_samples = sr * 5
        short_audio = audio_data[:max_samples]
        
        # 임시 파일로 5초 분량 오디오 저장
        temp_filename = "temp_gradio_5s.wav"
        sf.write(temp_filename, short_audio, sr)
        
        # --- Whisper 자동 자막 생성 ---
        if not ref_text or not ref_text.strip():
            print("원본 음성 대본(Reference Text)이 비어있어 Whisper 모델로 자동 추출합니다...")
            stt_result = stt_pipe(temp_filename, generate_kwargs={"language": "korean"})
            ref_text = stt_result["text"].strip()
            print(f"자동 추출된 대본: {ref_text}")
        
        print("음성을 복제하여 새로운 대사를 생성하는 중입니다...")
        
        # 4. Voice Clone 생성
        wavs, sr_out = model.generate_voice_clone(
            text=target_text,
            language="Korean", # 기본값을 한국어로 고정
            ref_audio=temp_filename,
            ref_text=ref_text
        )
        
        # 5. 결과물을 클라이언트에게 반환하기 위해 파일로 저장
        output_filename = "output_voice_clone_web.wav"
        sf.write(output_filename, wavs[0], sr_out)
        
        # 디버깅/디스크 정리를 위해 임시 파일 삭제
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        return output_filename, "🎉 성공적으로 음성이 복제되었습니다!"
        
    except Exception as e:
        return None, f"⚠️ 오류가 발생했습니다: {str(e)}"

# 6. Gradio Web UI 설계
with gr.Blocks(title="Qwen Voice Clone Web GUI") as demo:
    gr.Markdown("# 🎙️ Qwen Voice Clone Web UI")
    gr.Markdown("> 내 맥북에서 동작하는 로컬 AI 음성 복제 서버입니다. 5초 분량의 오디오만 있으면 목소리를 복제하여 새로운 대사를 생성할 수 있습니다.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. 🎤 오디오 업로드")
            audio_input = gr.Audio(type="filepath", label="Reference Audio (음성 파일 업로드 또는 직접 녹음)")
            
            gr.Markdown("### 2. 📝 실제 대사 입력 (선택사항)")
            with gr.Accordion("도움말 보기", open=False):
                gr.Markdown("Qwen Voice Clone 모델은 원본 음성의 내용을 인식해야 복제를 매끄럽게 수행합니다. 직접 입력하지 않으면 **Whisper AI가 자동으로 음성을 인식하여 대본을 작성**합니다.")
            ref_text_input = gr.Textbox(label="Reference Text (비워두면 자동 인식)", placeholder="빈칸으로 두시면 자동으로 인식됩니다. 혹시 직접 입력하고 싶다면 작성해 주세요.")
            
            gr.Markdown("### 3. ✨ 생성할 대사 입력")
            target_text_input = gr.Textbox(label="Target Text", placeholder="예: 와! 이렇게 웹 환경에서도 제 목소리가 완벽하게 복원되네요. 정말 신기합니다!", lines=3)
            
            generate_btn = gr.Button("🚀 복제된 음성 생성하기", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 🎧 생성 결과 재생 및 다운로드")
            audio_output = gr.Audio(label="Output Cloned Voice", interactive=False)
            status_output = gr.Textbox(label="상태 관리 메시지", interactive=False)
            
    # 클릭 이벤트를 함수에 바인딩
    generate_btn.click(
        fn=clone_voice,
        inputs=[audio_input, ref_text_input, target_text_input],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    # 0.0.0.0 (모든 네트워크 인터페이스)을 통해 맥북이 켜져있다면 다른 기기에서도 접속 가능하도록 세팅 (외부 공유 끄기)
    print("Starting Web Server... Access it via http://127.0.0.1:7860/ or your local network IP.")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
