// script.js
const monaImg    = document.getElementById('mona-lisa');
const video      = document.getElementById('video');
const canvasWarp = document.getElementById('canvasWarp');
const ctxWarp    = canvasWarp.getContext('2d');
const captureBtn = document.getElementById('captureBtn');
const saveBtn    = document.getElementById('saveBtn');

// 탐지 정확도/속도 균형 (threshold 낮춰 안정성↑)
const options    = new faceapi.TinyFaceDetectorOptions({
  inputSize: 320,
  scoreThreshold: 0.2
});

let monaPts      = [], triangles = [];
let isProcessing = false;

// 1. 초기화
async function init() {
  try {
    // 1-1) 모델 병렬 로드
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
      faceapi.nets.faceLandmark68Net.loadFromUri('./models')
    ]);

    // 1-2) Mona Lisa 이미지 완전 로드 대기
    if (!monaImg.complete) {
      await new Promise(res => monaImg.onload = res);
    }

    // 1-3) 캔버스 크기 설정
    canvasWarp.width  = monaImg.naturalWidth;
    canvasWarp.height = monaImg.naturalHeight;

    // 1-4) Mona Lisa 얼굴 랜드마크 + Delaunay
    const res = await faceapi
      .detectSingleFace(monaImg, options)
      .withFaceLandmarks();
    if (!res) throw new Error('Mona Lisa 얼굴 감지 실패');
    monaPts   = res.landmarks.positions.map(p => [p.x, p.y]);
    triangles = Delaunator.from(monaPts).triangles;

    // 1-5) 카메라 시작
    await startCamera();

    // 1-6) 비디오 메타데이터 준비 후 렌더 시작
    video.addEventListener('loadedmetadata', () => {
      // iOS용 playsinline 재보장
      video.playsinline = true;
      video.muted = true;
      video.play().catch(()=>{ /* 자동 재생 차단 시 무시 */ })
        .finally(() => requestAnimationFrame(render));
    });
  } catch (err) {
    console.error(err);
    alert('초기화 오류:\n' + err.message);
  }
}

// 2. 카메라 가져오기
function startCamera() {
  return navigator.mediaDevices
    .getUserMedia({ video: { facingMode: 'user' } })
    .then(stream => { video.srcObject = stream; })
    .catch(err => {
      console.error('카메라 접근 실패:', err);
      alert('카메라 권한을 허용해 주세요.');
    });
}

// 3. 매 프레임 얼굴 warp
async function render() {
  if (captureBtn.disabled) {
    // 캡처 모드일 땐 중단
    return;
  }

  // 중복 호출 방지
  if (isProcessing) {
    return requestAnimationFrame(render);
  }
  isProcessing = true;

  // 캔버스 초기화 (투명)
  ctxWarp.clearRect(0, 0, canvasWarp.width, canvasWarp.height);

  // 얼굴 탐지 (배열로 받고 첫 번째만)
  const detections = await faceapi
    .detectAllFaces(video, options)
    .withFaceLandmarks();
  if (detections.length > 0) {
    const face = detections[0];
    // 스케일 계산 (비디오 → 캔버스)
    const sx = canvasWarp.width  / video.videoWidth;
    const sy = canvasWarp.height / video.videoHeight;
    const vidPts = face.landmarks.positions.map(p => [p.x * sx, p.y * sy]);

    // Delaunay 삼각형별 warp
    for (let i = 0; i < triangles.length; i += 3) {
      const src = [
        vidPts[triangles[i]],
        vidPts[triangles[i+1]],
        vidPts[triangles[i+2]]
      ];
      const dst = [
        monaPts[triangles[i]],
        monaPts[triangles[i+1]],
        monaPts[triangles[i+2]]
      ];
      warp(src, dst);
    }
  }

  isProcessing = false;
  requestAnimationFrame(render);
}

// 4. Affine warp 헬퍼
function warp(src, dst) {
  const [[x0,y0],[x1,y1],[x2,y2]] = src;
  const [[u0,v0],[u1,v1],[u2,v2]] = dst;

  ctxWarp.save();
  ctxWarp.beginPath();
  ctxWarp.moveTo(u0,v0);
  ctxWarp.lineTo(u1,v1);
  ctxWarp.lineTo(u2,v2);
  ctxWarp.closePath();
  ctxWarp.clip();

  const m = getMatrix(x0,y0, x1,y1, x2,y2, u0,v0, u1,v1, u2,v2);
  ctxWarp.setTransform(m.a,m.b,m.c,m.d,m.e,m.f);
  ctxWarp.drawImage(video, 0, 0);
  ctxWarp.restore();
}

function getMatrix(x0,y0,x1,y1,x2,y2,u0,v0,u1,v1,u2,v2) {
  const denom = x0*(y1-y2) + x1*(y2-y0) + x2*(y0-y1);
  return {
    a: (u0*(y1-y2)+u1*(y2-y0)+u2*(y0-y1)) / denom,
    b: (v0*(y1-y2)+v1*(y2-y0)+v2*(y0-y1)) / denom,
    c: (u0*(x2-x1)+u1*(x0-x2)+u2*(x1-x0)) / denom,
    d: (v0*(x2-x1)+v1*(x0-x2)+v2*(x1-x0)) / denom,
    e: (u0*(x1*y2-x2*y1)+u1*(x2*y0-x0*y2)+u2*(x0*y1-x1*y0)) / denom,
    f: (v0*(x1*y2-x2*y1)+v1*(x2*y0-x0*y2)+v2*(x0*y1-x1*y0)) / denom
  };
}

// 5. 캡처 & 저장 흐름
captureBtn.addEventListener('click', () => {
  captureBtn.disabled = true;
  captureBtn.setAttribute('aria-disabled','true');
  saveBtn.style.display = 'inline-block';
  saveBtn.setAttribute('aria-disabled','false');
  // 스트림 중단
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
  }
});

saveBtn.addEventListener('click', () => {
  // warp 레이어를 Mona Lisa 위에 합성하여 Blob 생성
  const exportCanvas = document.createElement('canvas');
  exportCanvas.width  = canvasWarp.width;
  exportCanvas.height = canvasWarp.height;
  const ec = exportCanvas.getContext('2d');
  ec.drawImage(monaImg,    0, 0);
  ec.drawImage(canvasWarp, 0, 0);
  exportCanvas.toBlob(blob => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'mona_user_face.png';
    a.click();
    URL.revokeObjectURL(a.href);
  }, 'image/png');

  // 상태 복원
  saveBtn.style.display = 'none';
  saveBtn.setAttribute('aria-disabled','true');
  captureBtn.disabled = false;
  captureBtn.setAttribute('aria-disabled','false');
  startCamera(); // 재시작
});

window.addEventListener('load', init);
