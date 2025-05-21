// 전역 요소 셋업
const video       = document.getElementById('video');
const canvas      = document.getElementById('canvas');
const ctx         = canvas.getContext('2d');
const captureBtn  = document.getElementById('captureBtn');
const saveBtn     = document.getElementById('saveBtn');
const monaLisaImg = document.getElementById('mona-lisa');

// 모델 로드 후 비디오 스트림 시작
async function loadModels() {
  await faceapi.nets.tinyFaceDetector.loadFromUri('./models');
  await faceapi.nets.faceLandmark68Net.loadFromUri('./models');
  startVideo();
}

// 카메라 권한 요청 & 스트림 연결
function startVideo() {
  navigator.mediaDevices
    .getUserMedia({ video: {} })
    .then(stream => { video.srcObject = stream; })
    .catch(err => console.error('카메라 접근 실패:', err));
}

// 버튼 이벤트 설정
function setupButtons() {
  // 사진 찍기(캡처) 버튼
  captureBtn.addEventListener('click', async () => {
    await processFrame();
    saveBtn.style.display = 'inline-block';
  });
  // 저장하기 버튼
  saveBtn.addEventListener('click', () => {
    const dataUrl = canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = 'monalisa_result.png';
    link.click();
  });
}

// 한 번의 프레임 처리: 모나리자 워핑 & 그리기
async function processFrame() {
  // 캔버스 크기를 비디오 크기로 동기화
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;

  // 사용자 얼굴 랜드마크
  const userDet = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks();
  if (!userDet) {
    return console.warn('사용자 얼굴을 찾을 수 없습니다.');
  }
  const srcPts = userDet.landmarks.positions.map(p => [p.x, p.y]);

  // 모나리자 얼굴 랜드마크
  const monaDet = await faceapi
    .detectSingleFace(monaLisaImg, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks();
  if (!monaDet) {
    return console.warn('모나리자 얼굴을 찾을 수 없습니다.');
  }

  // 사전 계산된 모나리자 얼굴 바운딩 박스
  const faceBox = { x: 105, y: 79, width: 89, height: 89 };

  // dstPts: 박스 오프셋 제거 후 캔버스 크기로 스케일
  const dstPts = monaDet.landmarks.positions.map(p => [
    (p.x - faceBox.x) * canvas.width  / faceBox.width,
    (p.y - faceBox.y) * canvas.height / faceBox.height
  ]);

  // Delaunay 삼각 분할
  const delaunay  = Delaunator.from(dstPts);
  const triangles = delaunay.triangles;

  // 캔버스 초기화 & 모나리자 얼굴 영역만 그리기
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(
    monaLisaImg,
    faceBox.x, faceBox.y,
    faceBox.width, faceBox.height,
    0, 0,
    canvas.width, canvas.height
  );

  // 삼각형별 워핑
  for (let i = 0; i < triangles.length; i += 3) {
    const si = triangles[i], sj = triangles[i+1], sk = triangles[i+2];
    const srcTri = [ srcPts[si], srcPts[sj], srcPts[sk] ];
    const dstTri = [ dstPts[si], dstPts[sj], dstPts[sk] ];
    warpTriangle(video, ctx, srcTri, dstTri);
  }
}

// 삼각형 워핑 헬퍼
function warpTriangle(srcImg, dstCtx, srcTri, dstTri) {
  dstCtx.save();
  dstCtx.beginPath();
  dstTri.forEach(([x, y], idx) =>
    idx === 0 ? dstCtx.moveTo(x, y) : dstCtx.lineTo(x, y)
  );
  dstCtx.closePath();
  dstCtx.clip();

  const m = getAffineMatrix(srcTri, dstTri);
  dstCtx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
  dstCtx.drawImage(srcImg, 0, 0);
  dstCtx.restore();
}

// 어파인 행렬 계산
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

// 페이지 로드 시 초기화
window.addEventListener('load', () => {
  loadModels();
  setupButtons();
});
