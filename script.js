// script.js

const monaImg    = document.getElementById('mona-lisa');
const video      = document.getElementById('video');
const canvasWarp = document.getElementById('canvasWarp');
const ctxWarp    = canvasWarp.getContext('2d');
const captureBtn = document.getElementById('captureBtn');
const saveBtn    = document.getElementById('saveBtn');

const options    = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });
let monaPts      = [], triangles = [], isProcessing = false;

async function init() {
  // 1) 모델 로드
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('./models')
  ]);

  // 2) Mona Lisa 로드 대기 & 캔버스 크기 맞추기
  if (!monaImg.complete) await new Promise(r => monaImg.onload = r);
  canvasWarp.width  = monaImg.naturalWidth;
  canvasWarp.height = monaImg.naturalHeight;

  // 3) 얼굴 랜드마크 + Delaunay 분할
  const res = await faceapi.detectSingleFace(monaImg, options).withFaceLandmarks();
  if (!res) throw new Error('Mona Lisa 얼굴 감지 실패');
  monaPts   = res.landmarks.positions.map(p => [p.x, p.y]);
  triangles = Delaunator.from(monaPts).triangles;

  // 4) 카메라 시작
  await startCamera();

  // 5) 비디오가 재생될 때(render 호출 시점) 보장
  video.addEventListener('loadedmetadata', () => {
    video.play();
    requestAnimationFrame(render);
  });
}

function startCamera() {
  return navigator.mediaDevices
    .getUserMedia({ video: { facingMode: 'user' } })
    .then(stream => { video.srcObject = stream; });
}

async function render() {
  if (captureBtn.disabled) return;

  if (!isProcessing) {
    isProcessing = true;
    ctxWarp.clearRect(0, 0, canvasWarp.width, canvasWarp.height);

    // 1) 얼굴 탐지
    const result = await faceapi.detectSingleFace(video, options).withFaceLandmarks();
    console.log('detect result:', result);
    if (result) {
      // 2) 스케일 계산
      const sx = canvasWarp.width  / video.videoWidth;
      const sy = canvasWarp.height / video.videoHeight;
      const vidPts = result.landmarks.positions.map(p => [p.x * sx, p.y * sy]);

      // 3) warp
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
  }

  requestAnimationFrame(render);
}

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

// 캡처·저장 로직 (생략)… 그대로 두시면 됩니다.

window.addEventListener('load', init);
