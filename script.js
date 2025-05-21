// 전역 요소
const video       = document.getElementById('video');
const canvas      = document.getElementById('canvas');
const ctx         = canvas.getContext('2d');
const saveBtn     = document.getElementById('saveBtn');
const monaLisaImg = document.getElementById('mona-lisa');

// 1) 모델 로드 & 비디오 시작
async function loadModels() {
  await faceapi.nets.tinyFaceDetector.loadFromUri('./models');
  await faceapi.nets.faceLandmark68Net.loadFromUri('./models');
  startVideo();
}

function startVideo() {
  navigator.mediaDevices
    .getUserMedia({ video: { width: 640, height: 480 } })
    .then(stream => { video.srcObject = stream; })
    .catch(err => console.error('카메라 접근 실패:', err));
}

// 2) 비디오 재생 시 실시간 워핑 시작
video.addEventListener('play', () => {
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  setInterval(processFrame, 100);  // 100ms 마다 갱신
});

// 3) 매 프레임 얼굴 감지 & 워핑
async function processFrame() {
  // 3-1) 사용자 얼굴 랜드마크
  const userDet = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks();
  if (!userDet) return;
  const srcPts = userDet.landmarks.positions.map(p => [p.x, p.y]);

  // 3-2) 모나리자 얼굴 랜드마크
  const monaDet = await faceapi
    .detectSingleFace(monaLisaImg, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks();
  if (!monaDet) return;

  // 3-3) 사전 계산된 모나리자 얼굴 박스
  const faceBox = { x: 105, y: 79, width: 89, height: 89 };

  // 3-4) dstPts: 박스 오프셋 제거 → 캔버스 비율에 맞게 스케일
  const dstPts = monaDet.landmarks.positions.map(p => [
    (p.x - faceBox.x) * canvas.width  / faceBox.width,
    (p.y - faceBox.y) * canvas.height / faceBox.height
  ]);

  // 3-5) Delaunay 삼각 분할
  const delaunay  = Delaunator.from(dstPts);
  const triangles = delaunay.triangles;

  // 3-6) 캔버스에 전체 모나리자 그림 배경으로 그리기
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(monaLisaImg, 0, 0, canvas.width, canvas.height);

  // 3-7) 삼각형별 워핑
  for (let i = 0; i < triangles.length; i += 3) {
    const si = triangles[i], sj = triangles[i+1], sk = triangles[i+2];
    const srcTri = [ srcPts[si], srcPts[sj], srcPts[sk] ];
    const dstTri = [ dstPts[si], dstPts[sj], dstPts[sk] ];
    warpTriangle(video, ctx, srcTri, dstTri);
  }
}

// 4) 워핑 헬퍼 함수
function warpTriangle(src, dstCtx, srcTri, dstTri) {
  dstCtx.save();
  dstCtx.beginPath();
  dstTri.forEach(([x, y], idx) =>
    idx === 0 ? dstCtx.moveTo(x, y) : dstCtx.lineTo(x, y)
  );
  dstCtx.closePath();
  dstCtx.clip();

  const m = getAffineMatrix(srcTri, dstTri);
  dstCtx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
  dstCtx.drawImage(src, 0, 0);
  dstCtx.restore();
}

// 5) 어파인 변환 행렬 계산
function getAffineMatrix(src, dst) {
  const [[x0,y0],[x1,y1],[x2,y2]] = src;
  const [[u0,v0],[u1,v1],[u2,v2]] = dst;
  const denom = x0*(y1 - y2) + x1*(y2 - y0) + x2*(y0 - y1);

  return {
    a: (u0*(y1 - y2) + u1*(y2 - y0) + u2*(y0 - y1)) / denom,
    b: (v0*(y1 - y2) + v1*(y2 - y0) + v2*(y0 - y1)) / denom,
    c: (u0*(x2 - x1) + u1*(x0 - x2) + u2*(x1 - x0)) / denom,
    d: (v0*(x2 - x1) + v1*(x0 - x2) + v2*(x1 - x0)) / denom,
    e: (u0*(x1*y2 - x2*y1) + u1*(x2*y0 - x0*y2) + u2*(x0*y1 - x1*y0)) / denom,
    f: (v0*(x1*y2 - x2*y1) + v1*(x2*y0 - x0*y2) + v2*(x0*y1 - x1*y0)) / denom
  };
}

// 6) 저장하기 버튼
saveBtn.addEventListener('click', () => {
  const link = document.createElement('a');
  link.href = canvas.toDataURL('image/png');
  link.download = 'monalisa_swapped.png';
  link.click();
});

// 7) 초기화
window.addEventListener('load', loadModels);
