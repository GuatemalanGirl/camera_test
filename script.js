// script.js

// ——— 전역 변수 & 상수 ———
const video       = document.getElementById('video');
const canvas      = document.getElementById('canvas');
const ctx         = canvas.getContext('2d');
const monaLisaImg = document.getElementById('mona-lisa');
const captureBtn  = document.getElementById('captureBtn');
const saveBtn     = document.getElementById('saveBtn');

const options = new faceapi.TinyFaceDetectorOptions({
  inputSize: 320,
  scoreThreshold: 0.5
});

let isProcessing = false;
let monaLisaPts  = [];
let triangles    = [];

// ——— 1) 모델 로드 & 데이터 준비 ———
async function loadModelsAndData() {
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('./models')
  ]);

  if (!monaLisaImg.complete) {
    await new Promise(resolve => monaLisaImg.onload = resolve);
  }
  const res = await faceapi
    .detectSingleFace(monaLisaImg, options)
    .withFaceLandmarks();
  monaLisaPts = res.landmarks.positions.map(p => [p.x, p.y]);
  triangles   = Delaunator.from(monaLisaPts).triangles;

  startVideo();
}

// ——— 2) 카메라 시작 ———
function startVideo() {
  navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, facingMode: 'user' }
  }).then(stream => {
    video.srcObject = stream;
  }).catch(err => {
    console.error('카메라 접근 실패:', err);
    alert('카메라를 사용할 수 없습니다.');
  });
}

// ——— 3) 렌더 루프 ———
video.addEventListener('play', () => {
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  requestAnimationFrame(renderFrame);
});

async function renderFrame() {
  // “사진 찍기” 누르면 루프 종료
  if (captureBtn.disabled) return;

  if (isProcessing) {
    requestAnimationFrame(renderFrame);
    return;
  }
  isProcessing = true;

  // 1) 항상 비디오 배경부터
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  try {
    // 2) 얼굴 검출 + 랜드마크
    const res = await faceapi
      .detectSingleFace(video, options)
      .withFaceLandmarks();
    if (res) {
      const dstPts = res.landmarks.positions.map(p => [p.x, p.y]);
      // 3) Delaunay 워핑
      for (let i = 0; i < triangles.length; i += 3) {
        const srcTri = [
          monaLisaPts[triangles[i]],
          monaLisaPts[triangles[i+1]],
          monaLisaPts[triangles[i+2]]
        ];
        const dstTri = [
          dstPts[triangles[i]],
          dstPts[triangles[i+1]],
          dstPts[triangles[i+2]]
        ];
        warpTriangle(monaLisaImg, ctx, srcTri, dstTri);
      }
    }
  } catch (e) {
    console.error('프레임 처리 오류:', e);
  }

  isProcessing = false;
  requestAnimationFrame(renderFrame);
}

// ——— 4) 워핑 헬퍼 ———
function warpTriangle(img, context, src, dst) {
  context.save();
  context.beginPath();
  context.moveTo(...dst[0]);
  context.lineTo(...dst[1]);
  context.lineTo(...dst[2]);
  context.closePath();
  context.clip();

  const m = getAffineMatrix(...src[0], ...src[1], ...src[2],
                            ...dst[0], ...dst[1], ...dst[2]);
  context.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
  context.drawImage(img, 0, 0);
  context.restore();
}

function getAffineMatrix(x0, y0, x1, y1, x2, y2,
                         u0, v0, u1, v1, u2, v2) {
  const den = x0*(y1-y2) + x1*(y2-y0) + x2*(y0-y1);
  const a = (u0*(y1-y2) + u1*(y2-y0) + u2*(y0-y1)) / den;
  const b = (v0*(y1-y2) + v1*(y2-y0) + v2*(y0-y1)) / den;
  const c = (u0*(x2-x1) + u1*(x0-x2) + u2*(x1-x0)) / den;
  const d = (v0*(x2-x1) + v1*(x0-x2) + v2*(x1-x0)) / den;
  const e = (u0*(x1*y2-x2*y1) + u1*(x2*y0-x0*y2) + u2*(x0*y1-x1*y0)) / den;
  const f = (v0*(x1*y2-x2*y1) + v1*(x2*y0-x0*y2) + v2*(x0*y1-x1*y0)) / den;
  return { a, b, c, d, e, f };
}

// ——— 5) “사진 찍기” 클릭 ———
captureBtn.addEventListener('click', () => {
  captureBtn.disabled = true;
  saveBtn.style.display = 'inline-block';

  // 스트림은 저장 후 중단할 거예요
});

// ——— 6) “저장하기” 클릭 ———
saveBtn.addEventListener('click', () => {
  // 1) 합성용 캔버스 생성
  const exportCanvas = document.createElement('canvas');
  exportCanvas.width  = video.videoWidth;
  exportCanvas.height = video.videoHeight;
  const ec = exportCanvas.getContext('2d');

  // 2) 합성 순서: 비디오 → 오버레이
  ec.drawImage(video, 0, 0, exportCanvas.width, exportCanvas.height);
  ec.drawImage(canvas, 0, 0, exportCanvas.width, exportCanvas.height);

  // 3) iOS 대응: Web Share API 우선, 아니면 새 탭 열기
  exportCanvas.toBlob(async blob => {
    const file = new File([blob], 'capture.png', { type: 'image/png' });
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      try {
        await navigator.share({ files: [file] });
      } catch (err) {
        console.error('공유 실패:', err);
        window.open(URL.createObjectURL(blob), '_blank');
      }
    } else {
      window.open(URL.createObjectURL(blob), '_blank');
    }

    // 4) 스트림 중단 & 버튼 상태 복구
    if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
    captureBtn.disabled = false;
    saveBtn.style.display = 'none';
  }, 'image/png');
});

// ——— 7) 페이지 로드 시 실행 ———
window.addEventListener('load', loadModelsAndData);
