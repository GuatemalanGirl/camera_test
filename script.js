// script.js

// ——————————————————————————————
// 전역 변수 셋업
// ——————————————————————————————
const video        = document.getElementById('video');
const canvas       = document.getElementById('canvas');
const ctx          = canvas.getContext('2d');
const downloadLink = document.getElementById('downloadLink');
const monaLisaImg  = document.getElementById('mona-lisa');

// ——————————————————————————————
// 1) 모델 로드 & 비디오 시작
// ——————————————————————————————
Promise.all([
  faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('./models')
]).then(startVideo);

function startVideo() {
  navigator.mediaDevices
    .getUserMedia({ video: { width: 320, height: 240 } })
    .then(stream => { video.srcObject = stream; })
    .catch(err => console.error(err));
}

video.addEventListener('play', () => {
  // 간단히 200ms 마다 처리
  setInterval(processFrame, 200);
});

// ——————————————————————————————
// 2) 비디오 프레임마다 워핑 처리
// ——————————————————————————————
video.addEventListener('play', () => {
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  // 100ms마다 화면 갱신
  setInterval(processFrame, 100);
});

async function processFrame() {
  // 2-1) 사용자 얼굴 랜드마크 추출
  const userDet = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks();
  if (!userDet) return console.warn('사용자 얼굴을 못 찾았어요.');

  // 2-2) 모나리자 얼굴 랜드마크 추출
  const monaDet = await faceapi
    .detectSingleFace(monaLisaImg, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks();
  if (!monaDet) return console.warn('모나리자 얼굴을 못 찾았어요.');

  // 2-3) 랜드마크 좌표만 뽑아서 배열로
  const srcPts = userDet.landmarks.positions.map(p => [p.x, p.y]);
  const dstPts = monaDet.landmarks.positions.map(p => [p.x, p.y]);

  // 2-4) Delaunay 삼각 분할
  const delaunay   = Delaunator.from(dstPts);
  const triangles  = delaunay.triangles;

  // 2-5) 모나리자 원본 캔버스에 그리기
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(monaLisaImg, 0, 0, canvas.width, canvas.height);

  // 2-6) 각 삼각형별로 비디오 프레임 워핑
  for (let i = 0; i < triangles.length; i += 3) {
    const si     = triangles[i],
          sj     = triangles[i+1],
          sk     = triangles[i+2];
    const srcTri = [ srcPts[si], srcPts[sj], srcPts[sk] ];
    const dstTri = [ dstPts[si], dstPts[sj], dstPts[sk] ];
    warpTriangle(video, ctx, srcTri, dstTri);
  }

  // 2-7) 다운로드 링크 활성화
  downloadLink.href           = canvas.toDataURL('image/png');
  downloadLink.style.display  = 'inline';
}

// ——————————————————————————————
// 3) 삼각형 워핑 함수
// ——————————————————————————————
function warpTriangle(srcImg, dstCtx, srcTri, dstTri) {
  dstCtx.save();

  // 3-1) 목적 삼각형 영역 클리핑
  dstCtx.beginPath();
  dstTri.forEach(([x, y], idx) =>
    idx === 0 ? dstCtx.moveTo(x, y) : dstCtx.lineTo(x, y)
  );
  dstCtx.closePath();
  dstCtx.clip();

  // 3-2) 어파인 변환 행렬 계산
  const m = getAffineMatrix(srcTri, dstTri);
  dstCtx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);

  // 3-3) 원본(비디오) 그리기
  dstCtx.drawImage(srcImg, 0, 0);

  dstCtx.restore();
}

// ——————————————————————————————
// 4) 어파인 행렬 구하는 함수
// ——————————————————————————————
function getAffineMatrix(src, dst) {
  const [[x0,y0],[x1,y1],[x2,y2]] = src;
  const [[u0,v0],[u1,v1],[u2,v2]] = dst;

  const denom = x0*(y1 - y2) + x1*(y2 - y0) + x2*(y0 - y1);

  const a = (u0*(y1 - y2) + u1*(y2 - y0) + u2*(y0 - y1)) / denom;
  const b = (v0*(y1 - y2) + v1*(y2 - y0) + v2*(y0 - y1)) / denom;
  const c = (u0*(x2 - x1) + u1*(x0 - x2) + u2*(x1 - x0)) / denom;
  const d = (v0*(x2 - x1) + v1*(x0 - x2) + v2*(x1 - x0)) / denom;
  const e = (u0*(x1*y2 - x2*y1) + u1*(x2*y0 - x0*y2) + u2*(x0*y1 - x1*y0)) / denom;
  const f = (v0*(x1*y2 - x2*y1) + v1*(x2*y0 - x0*y2) + v2*(x0*y1 - x1*y0)) / denom;

  return { a, b, c, d, e, f };
}
