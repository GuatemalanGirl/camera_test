// script.js
const monaImg = document.getElementById('mona-lisa');
const canvas  = document.getElementById('canvas');
const video   = document.getElementById('video');
const startBtn= document.getElementById('start');
const snapBtn = document.getElementById('snap');
const saveBtn = document.getElementById('save');
const ctx     = canvas.getContext('2d');

// TinyFaceDetector 옵션 (실시간 성능과 감지율 균형)
const options = new faceapi.TinyFaceDetectorOptions({
  inputSize: 128,
  scoreThreshold: 0.4
});

let monaBox;    // Mona Lisa 얼굴 영역
let streamOn = false;

// 1) 모델과 Mona Lisa 얼굴 위치 미리 로드
async function preload() {
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('./models')
  ]);
  if (!monaImg.complete) {
    await new Promise(r => monaImg.onload = r);
  }
  canvas.width  = monaImg.naturalWidth;
  canvas.height = monaImg.naturalHeight;

  const det = await faceapi.detectSingleFace(monaImg, options);
  if (!det) {
    alert('Mona Lisa 얼굴을 찾을 수 없습니다.');
    throw new Error('Face not detected in Mona Lisa');
  }
  monaBox = det.box;
}
preload().catch(e => console.error(e));

// 2) “카메라 시작” 클릭 → 권한 요청 · 스트림 연결
startBtn.addEventListener('click', async () => {
  if (streamOn) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user' }
    });
    video.srcObject = stream;
    streamOn = true;
    snapBtn.disabled = false;

    // video가 play되면 렌더 루프 시작
    video.addEventListener('play', () => render(), { once: true });
  } catch (err) {
    console.error(err);
    alert('카메라 권한을 허용해 주세요.');
  }
});

// 3) 실시간 합성 렌더 함수
async function render() {
  if (!streamOn) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // Mona Lisa 전체
  ctx.drawImage(monaImg, 0, 0, canvas.width, canvas.height);

  // 사용자 얼굴 탐지
  const det = await faceapi
    .detectSingleFace(video, options)
    .withFaceLandmarks();

  if (det) {
    const { x: ux, y: uy, width: uw, height: uh } = det.detection.box;
    const { x: mx, y: my, width: mw, height: mh } = monaBox;
    // video(source) → Mona Lisa 얼굴 영역(destination)
    ctx.drawImage(video, ux, uy, uw, uh, mx, my, mw, mh);
  }

  requestAnimationFrame(render);
}

// 4) “사진 찍기” 클릭 → 스트림 중지 · 캔버스 고정
snapBtn.addEventListener('click', () => {
  if (!streamOn) return;
  streamOn = false;
  video.pause();
  video.srcObject.getTracks().forEach(t => t.stop());
  saveBtn.disabled = false;
});

// 5) “저장하기” 클릭 → PNG 다운로드
saveBtn.addEventListener('click', () => {
  const link = document.createElement('a');
  link.download = 'mona_face_swap.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
});
