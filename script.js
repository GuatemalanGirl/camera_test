// script.js
const monaImg    = document.getElementById('mona-lisa');
const video      = document.getElementById('video');
const canvas     = document.getElementById('canvas');
const ctx        = canvas.getContext('2d');
const captureBtn = document.getElementById('captureBtn');
const saveBtn    = document.getElementById('saveBtn');

// TinyFaceDetector 옵션 (threshold 낮춰 안정성↑)
const options = new faceapi.TinyFaceDetectorOptions({
  inputSize: 320,
  scoreThreshold: 0.2
});

let monaBox;  // MonaLisa 얼굴영역 박스

async function init() {
  // 1) 모델 로드
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('./models')
  ]);

  // 2) MonaLisa 완전 로드 대기
  if (!monaImg.complete) {
    await new Promise(res => monaImg.onload = res);
  }

  // 3) 캔버스 크기 맞춤
  canvas.width  = monaImg.naturalWidth;
  canvas.height = monaImg.naturalHeight;

  // 4) MonaLisa 얼굴 박스 계산 (한 번만)
  const det = await faceapi.detectSingleFace(monaImg, options);
  if (!det) {
    alert('Mona Lisa 얼굴을 찾을 수 없습니다.');
    return;
  }
  // 바로 det.box 를 사용합니다.
  monaBox = det.box; // { x, y, width, height }

  // 5) 카메라 시작
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (e) {
    alert('카메라 권한이 필요합니다.');
    return;
  }

  // 6) 비디오 준비 후 렌더 루프 시작
  video.addEventListener('loadedmetadata', () => {
    video.play().catch(()=>{}).finally(() => requestAnimationFrame(render));
  });
}

// 매 프레임마다 MonaLisa → 사용자 얼굴 매핑
async function render() {
  if (captureBtn.disabled) return;

  // 1) MonaLisa 배경
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(monaImg, 0, 0, canvas.width, canvas.height);

  // 2) 사용자 얼굴 탐지
  const userDet = await faceapi.detectSingleFace(video, options);
  if (userDet) {
    const ub = userDet.box; 
    // drawImage(video, sx, sy, sw, sh, dx, dy, dw, dh)
    ctx.drawImage(
      video,
      ub.x, ub.y, ub.width, ub.height,
      monaBox.x, monaBox.y, monaBox.width, monaBox.height
    );
  }

  requestAnimationFrame(render);
}

// “사진 찍기” → 스트림 중단
captureBtn.addEventListener('click', () => {
  captureBtn.disabled = true;
  saveBtn.style.display = 'inline-block';
  video.srcObject.getTracks().forEach(t => t.stop());
});

// “저장하기” → 캔버스 PNG 다운로드
saveBtn.addEventListener('click', () => {
  canvas.toBlob(blob => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'mona_user_face.png';
    a.click();
    URL.revokeObjectURL(a.href);
  });
  saveBtn.style.display = 'none';
  captureBtn.disabled = false;
  // 재시작
  video.play().catch(()=>{}).finally(() => requestAnimationFrame(render));
});

window.addEventListener('load', init);
