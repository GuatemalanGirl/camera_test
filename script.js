const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const MONALISA_FACE = {
  x: 190, y: 140, width: 140, height: 170 // 합성될 얼굴 위치
};

const monalisaImg = new Image();
monalisaImg.src = 'monalisa.jpg';

async function setup() {
  // 모델 로딩
  await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
  await faceapi.nets.faceLandmark68Net.loadFromUri('/models');

  // 카메라 연결
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;

  video.addEventListener('play', () => {
    setInterval(processFrame, 200);
  });
}

async function processFrame() {
  if (video.readyState < 2) return;

  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  // 모나리자 배경 그리기
  ctx.drawImage(monalisaImg, 0, 0, canvas.width, canvas.height);

  // 얼굴 인식
  const detection = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks();

  if (detection) {
    const box = detection.detection.box;

    // 사용자 얼굴 crop
    const faceCanvas = document.createElement('canvas');
    faceCanvas.width = box.width;
    faceCanvas.height = box.height;
    faceCanvas.getContext('2d').drawImage(
      video,
      box.x, box.y, box.width, box.height,
      0, 0, box.width, box.height
    );

    // 모나리자 얼굴 위치에 맞게 합성
    ctx.drawImage(
      faceCanvas,
      0, 0, box.width, box.height,
      MONALISA_FACE.x, MONALISA_FACE.y, MONALISA_FACE.width, MONALISA_FACE.height
    );
  }
}

const captureBtn = document.getElementById('captureBtn');
const downloadBtn = document.getElementById('downloadBtn');

let lastCaptured = null;

captureBtn.addEventListener('click', async () => {
  await processFrame(); // 현재 프레임 처리
  lastCaptured = canvas.toDataURL('image/png'); // 이미지 저장
  alert('📸 촬영 완료! 다운로드할 수 있어요.');
});

downloadBtn.addEventListener('click', () => {
  if (!lastCaptured) {
    alert('먼저 사진을 촬영하세요!');
    return;
  }
  const link = document.createElement('a');
  link.href = lastCaptured;
  link.download = 'monalisa_swap.png';
  link.click();
});

setup();
