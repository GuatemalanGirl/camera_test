const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

const MONALISA_FACE = {
  x: 190, y: 140, width: 140, height: 170 // 합성될 얼굴 위치
};

const monalisaImg = new Image();
monalisaImg.src = './monalisa.jpg'; // 현재 경로 기준

async function setup() {
  await faceapi.nets.tinyFaceDetector.loadFromUri('./models');
  await faceapi.nets.faceLandmark68Net.loadFromUri('./models');

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

  ctx.drawImage(monalisaImg, 0, 0, canvas.width, canvas.height);

  const detection = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks();

  if (detection) {
    const box = detection.detection.box;

    const faceCanvas = document.createElement('canvas');
    faceCanvas.width = box.width;
    faceCanvas.height = box.height;
    faceCanvas.getContext('2d').drawImage(
      video,
      box.x, box.y, box.width, box.height,
      0, 0, box.width, box.height
    );

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
  await processFrame();
  lastCaptured = canvas.toDataURL('image/png');
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
