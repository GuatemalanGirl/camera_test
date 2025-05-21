// script.js

// —— 전역 변수 & 상수 정의 ——
const video       = document.getElementById('video');
const canvas      = document.getElementById('canvas');
const ctx         = canvas.getContext('2d');
const monaLisaImg = document.getElementById('mona-lisa');
const captureBtn  = document.getElementById('captureBtn');
const saveBtn     = document.getElementById('saveBtn');

// 얼굴 인식 옵션 (TinyFaceDetector)
const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });

let isProcessing = false;       // 중복 프레임 처리 방지 플래그
let monaLisaPts  = [];          // 모나리자 랜드마크 좌표
let triangles    = [];          // Delaunay 삼각형 인덱스

// —— 1) 모델 로드 & 모나리자 데이터 준비 ——
async function loadModelsAndData() {
  try {
    // 병렬로 모델 로드
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
      faceapi.nets.faceLandmark68Net.loadFromUri('./models')
    ]);

    // 이미지 완전 로드 대기
    if (!monaLisaImg.complete) {
      await new Promise(resolve => monaLisaImg.onload = resolve);
    }
    await prepareMonaLisa();    // 모나리자 랜드마크 감지 + Delaunay 계산
    startVideo();               // 카메라 시작
  } catch (err) {
    console.error('모델/데이터 로드 실패:', err);
    alert('모델을 불러오는 중 오류가 발생했습니다. 네트워크를 확인해주세요.');
  }
}

// —— 1-1) 모나리자 얼굴 탐지 및 삼각 분할 ——
async function prepareMonaLisa() {
  try {
    const res = await faceapi
      .detectSingleFace(monaLisaImg, options)
      .withFaceLandmarks();
    if (!res) throw new Error('Mona Lisa 얼굴 감지 실패');
    monaLisaPts = res.landmarks.positions.map(p => [p.x, p.y]);
    triangles   = Delaunator.from(monaLisaPts).triangles;
  } catch (err) {
    console.error('MonaLisa 처리 오류:', err);
    alert('Mona Lisa 이미지에서 얼굴을 감지하지 못했습니다.');
  }
}

// —— 2) 카메라 시작 ——
function startVideo() {
  navigator.mediaDevices
    .getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } })
    .then(stream => {
      video.srcObject = stream;
    })
    .catch(err => {
      console.error('카메라 접근 실패:', err);
      alert('카메라를 사용할 수 없습니다.');
    });
}

// —— 3) 프레임 렌더링 (requestAnimationFrame) ——
video.addEventListener('play', () => {
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  renderFrame();
});

async function renderFrame() {
  // 캡처 모드라면 계속 그리지 않음
  if (captureBtn.disabled) return;

  if (isProcessing) {
    requestAnimationFrame(renderFrame);
    return;
  }
  isProcessing = true;

  try {
    const res = await faceapi
      .detectSingleFace(video, options)
      .withFaceLandmarks();
    if (res && monaLisaPts.length) {
      const dstPts = res.landmarks.positions.map(p => [p.x, p.y]);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 미리 계산한 triangles 재사용
      for (let i = 0; i < triangles.length; i += 3) {
        const srcTri = [
          monaLisaPts[triangles[i]],
          monaLisaPts[triangles[i + 1]],
          monaLisaPts[triangles[i + 2]],
        ];
        const dstTri = [
          dstPts[triangles[i]],
          dstPts[triangles[i + 1]],
          dstPts[triangles[i + 2]],
        ];
        warpTriangle(monaLisaImg, ctx, srcTri, dstTri);
      }
    }
  } catch (err) {
    console.error('프레임 처리 오류:', err);
  }

  isProcessing = false;
  requestAnimationFrame(renderFrame);
}

// —— 4) 삼각형 워핑 헬퍼 함수들 ——
function warpTriangle(img, context, src, dst) {
  const [[x0, y0], [x1, y1], [x2, y2]] = src;
  const [[u0, v0], [u1, v1], [u2, v2]] = dst;

  context.save();
  context.beginPath();
  context.moveTo(u0, v0);
  context.lineTo(u1, v1);
  context.lineTo(u2, v2);
  context.closePath();
  context.clip();

  const m = getAffineMatrix([x0, y0], [x1, y1], [x2, y2], [u0, v0], [u1, v1], [u2, v2]);
  context.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
  context.drawImage(img, 0, 0);
  context.restore();
}

function getAffineMatrix(p0, p1, p2, q0, q1, q2) {
  const [x0, y0] = p0, [x1, y1] = p1, [x2, y2] = p2;
  const [u0, v0] = q0, [u1, v1] = q1, [u2, v2] = q2;
  const denom = x0*(y1 - y2) + x1*(y2 - y0) + x2*(y0 - y1);

  const a = (u0*(y1 - y2) + u1*(y2 - y0) + u2*(y0 - y1)) / denom;
  const b = (v0*(y1 - y2) + v1*(y2 - y0) + v2*(y0 - y1)) / denom;
  const c = (u0*(x2 - x1) + u1*(x0 - x2) + u2*(x1 - x0)) / denom;
  const d = (v0*(x2 - x1) + v1*(x0 - x2) + v2*(x1 - x0)) / denom;
  const e = (u0*(x1*y2 - x2*y1) + u1*(x2*y0 - x0*y2) + u2*(x0*y1 - x1*y0)) / denom;
  const f = (v0*(x1*y2 - x2*y1) + v1*(x2*y0 - x0*y2) + v2*(x0*y1 - x1*y0)) / denom;

  return { a, b, c, d, e, f };
}

// —— 5) “사진 찍기” 클릭 ——
captureBtn.addEventListener('click', () => {
  captureBtn.disabled = true;
  captureBtn.setAttribute('aria-disabled', 'true');
  saveBtn.style.display = 'inline-block';
  saveBtn.setAttribute('aria-disabled', 'false');

  // 비디오 스트림 완전 중단
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(t => t.stop());
  }
});

// —— 6) “저장하기” 클릭 ——
saveBtn.addEventListener('click', () => {
  // 1) 내보낼 캔버스 생성
  const exportCanvas = document.createElement('canvas');
  exportCanvas.width  = canvas.width;
  exportCanvas.height = canvas.height;
  const ec = exportCanvas.getContext('2d');

  // 2) 카메라 비디오 프레임을 가장 먼저 그린다
  ec.drawImage(video, 0, 0, exportCanvas.width, exportCanvas.height);

  // 3) 그 위에 오버레이(얼굴 워핑된 캔버스)를 올린다
  ec.drawImage(canvas, 0, 0, exportCanvas.width, exportCanvas.height);

  // 4) Blob 으로 변환 → 다운로드
  exportCanvas.toBlob(blob => {
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'full_composite.png';
    link.click();
    URL.revokeObjectURL(link.href);

    // 5) (선택) 스트림 중단: **반드시** drawImage 호출 뒤에!
    // video.srcObject.getTracks().forEach(t => t.stop());
  }, 'image/png');
});

// —— 7) 페이지 로드 시 실행 ——
window.addEventListener('load', loadModelsAndData);
