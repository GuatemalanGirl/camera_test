// script.js

const video       = document.getElementById('video');
const canvas      = document.getElementById('canvas');
const ctx         = canvas.getContext('2d');
const monaLisaImg = document.getElementById('mona-lisa');
const captureBtn  = document.getElementById('captureBtn');
const saveBtn     = document.getElementById('saveBtn');

const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });
let monaLisaPts  = [], triangles = [];
let isProcessing = false;

// 1) 모델 로드 및 모나리자 데이터 준비
async function init() {
  try {
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
      faceapi.nets.faceLandmark68Net.loadFromUri('./models')
    ]);

    if (!monaLisaImg.complete) {
      await new Promise(res => monaLisaImg.onload = res);
    }
    prepareCanvas();
    await prepareMonaLisa();
    startVideo();
  } catch (e) {
    console.error(e);
    alert('초기화 오류 – 모델 또는 이미지 로드 실패');
  }
}

// 캔버스 크기를 모나리자 원본 크기로 설정
function prepareCanvas() {
  canvas.width  = monaLisaImg.naturalWidth;
  canvas.height = monaLisaImg.naturalHeight;
}

// 모나리자 얼굴 랜드마크 + Delaunay 계산
async function prepareMonaLisa() {
  const res = await faceapi.detectSingleFace(monaLisaImg, options).withFaceLandmarks();
  if (!res) throw new Error('Mona Lisa 얼굴 감지 실패');
  monaLisaPts = res.landmarks.positions.map(p => [p.x, p.y]);
  triangles   = Delaunator.from(monaLisaPts).triangles;
}

// 2) 카메라 스트림 시작
function startVideo() {
  navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } })
    .then(stream => { video.srcObject = stream; })
    .catch(err => { console.error(err); alert('카메라 접근 불가'); });
}

// 3) 비디오가 재생되면 렌더링 루프 시작
video.addEventListener('play', () => {
  const scaleX = monaLisaImg.naturalWidth  / video.videoWidth;
  const scaleY = monaLisaImg.naturalHeight / video.videoHeight;

  async function render() {
    if (captureBtn.disabled) return;
    if (isProcessing) {
      requestAnimationFrame(render);
      return;
    }
    isProcessing = true;

    // 1) 베이스로 모나리자 그림 그리기
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(monaLisaImg, 0,0, canvas.width, canvas.height);

    // 2) 사용자 얼굴 감지
    const res = await faceapi.detectSingleFace(video, options).withFaceLandmarks();
    if (res) {
      const vidPts = res.landmarks.positions.map(p => [p.x * scaleX, p.y * scaleY]);

      // 3) 삼각형마다 사용자 얼굴 조각을 모나리자 얼굴 위치로 warp
      for (let i = 0; i < triangles.length; i += 3) {
        const srcTri = [
          vidPts[triangles[i]],
          vidPts[triangles[i+1]],
          vidPts[triangles[i+2]],
        ];
        const dstTri = [
          monaLisaPts[triangles[i]],
          monaLisaPts[triangles[i+1]],
          monaLisaPts[triangles[i+2]],
        ];
        warpTriangle(video, ctx, srcTri, dstTri);
      }
    }

    isProcessing = false;
    requestAnimationFrame(render);
  }

  render();
});

// 삼각형 워핑 헬퍼
function warpTriangle(srcEl, ctx, src, dst) {
  const [[x0,y0],[x1,y1],[x2,y2]] = src;
  const [[u0,v0],[u1,v1],[u2,v2]] = dst;

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(u0, v0);
  ctx.lineTo(u1, v1);
  ctx.lineTo(u2, v2);
  ctx.closePath();
  ctx.clip();

  const m = getAffineMatrix(x0,y0, x1,y1, x2,y2, u0,v0, u1,v1, u2,v2);
  ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
  ctx.drawImage(srcEl, 0, 0);
  ctx.restore();
}

function getAffineMatrix(x0,y0, x1,y1, x2,y2, u0,v0, u1,v1, u2,v2) {
  const denom = x0*(y1-y2) + x1*(y2-y0) + x2*(y0-y1);
  const a = (u0*(y1-y2) + u1*(y2-y0) + u2*(y0-y1)) / denom;
  const b = (v0*(y1-y2) + v1*(y2-y0) + v2*(y0-y1)) / denom;
  const c = (u0*(x2-x1) + u1*(x0-x2) + u2*(x1-x0)) / denom;
  const d = (v0*(x2-x1) + v1*(x0-x2) + v2*(x1-x0)) / denom;
  const e = (u0*(x1*y2-x2*y1) + u1*(x2*y0-x0*y2) + u2*(x0*y1-x1*y0)) / denom;
  const f = (v0*(x1*y2-x2*y1) + v1*(x2*y0-x0*y2) + v2*(x0*y1-x1*y0)) / denom;
  return { a,b,c,d,e,f };
}

// 4) 캡처/저장 로직 (이전과 동일)
captureBtn.addEventListener('click', () => {
  captureBtn.disabled = true;
  captureBtn.setAttribute('aria-disabled','true');
  saveBtn.style.display = 'inline-block';
  saveBtn.setAttribute('aria-disabled','false');
  if (video.srcObject) video.srcObject.getTracks().forEach(t => t.stop());
});

saveBtn.addEventListener('click', () => {
  const ex = document.createElement('canvas');
  ex.width  = canvas.width;
  ex.height = canvas.height;
  const ec = ex.getContext('2d');
  ec.drawImage(canvas, 0,0);
  ex.toBlob(blob => {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'monalisa_selfie.png';
    link.click();
    URL.revokeObjectURL(link.href);
  }, 'image/png');

  saveBtn.style.display = 'none';
  saveBtn.setAttribute('aria-disabled','true');
  captureBtn.disabled = false;
  captureBtn.setAttribute('aria-disabled','false');
  startVideo();
});

window.addEventListener('load', init);
