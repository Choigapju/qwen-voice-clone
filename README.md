# Qwen Voice Clone Web Application

이 프로젝트는 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 모델을 기반으로 한 음성 복제(Voice Clone) 웹 애플리케이션입니다.
`Gradio`를 사용하여 간편하게 오디오를 업로드하고, 원하는 대사를 입력하면 복제된 목소리가 생성됩니다.

## 기능 (Features)
- 🎙️ **오디오 업로드 및 녹음**: 파일 업로드나 마이크를 통해 원본 음성 제공
- ✂️ **자동 5초 추출**: 업로드한 오디오에서 처음 5초 분량만 추출하여 모델에 전달
- 📝 **자동 대본 생성(Whisper AI)**: 번거롭게 대본을 치지 않아도 Whisper 최신 모델을 통해 앞부분 5초를 자동으로 인식
- 🗣️ **목소리 복제**: 원본 음성의 목소리 특징을 완벽하게 따라하여 새로운 대사 생성
- 🚀 **로컬 네트워크 웹 서빙**: Macbook 등의 로컬 기기에 서버를 켜두고 동일 네트워크 기기에서 접속 가능

## 설치 방법 (Installation)

1. **저장소 클론 및 가상 환경 준비**
   ```bash
   git clone <your-repo-url>
   cd Qwen
   python -m venv qwen-env
   source qwen-env/bin/activate
   ```

2. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```
   > 💡 Flash Attention이 지원되는 GPU를 사용한다면 `pip install flash-attn`을 권장합니다.

## 실행 방법 (Usage)

웹 서버를 실행하려면 아래 명령어를 입력하세요:
```bash
python app.py
```

성공적으로 실행되면, 터미널에 로컬 주소가 출력됩니다 (예: `http://0.0.0.0:7860` 또는 `http://127.0.0.1:7860`). 웹 브라우저를 열고 해당 주소로 접속하세요.

### Voice Clone 사용 팁!
이제 번거롭게 원본 대사를 치지 않아도 됩니다! 오디오 파일만 업로드하고 **Reference Text** 칸을 비워두시면, 내장된 **Whisper AI** 가 처음 5초 동안의 음성을 자동으로 인식하여 대본을 작성해 줍니다. (물론 직접 수정해서 입력하실 수도 있습니다.)

---

*본 프로젝트는 Mac(MPS GPU) 환경을 기본으로 호환되도록 구성되었습니다.*
