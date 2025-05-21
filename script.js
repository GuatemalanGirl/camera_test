// script.js
const monaImg = document.getElementById('mona-lisa');
const canvas  = document.getElementById('canvas');
const video   = document.getElementById('video');
const startBtn= document.getElementById('start');
const snapBtn = document.getElementById('snap');
const saveBtn = document.getElementById('save');
const ctx     = canvas.getContext('2d');

const options = new faceapi.TinyFaceDetectorOptions({ inputSize:128, scoreThreshold:0.5 });
let monaBox;

// 모델과 이미지 로드
async function preload() {
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('./models')
  ]);
  await new Promise(r => monaImg.complete ? r() : (monaImg.onload = r));
  canvas.width = monaImg.naturalWidth;
  canvas.height= monaImg.naturalHeight;
  const det = await faceapi.detectSingleFace(monaImg, options);
  if (!det) throw new Error('모나리자 얼굴을 찾을 수 없습니다.');
  monaBox = det.box;
}

preload().catch(e => {
  alert(e.message);
  console.error(e);
});

// “카메라 시작” 버튼
startBtn.addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode:'user' } });
    video.srcObject = stream;
    render();
  } catch (err) {
    alert('카메라 권한이 필요합니다.');
    console.error(err);
  }
});

// 렌더 함수
async function render() {
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(monaImg, 0,0,canvas.width,canvas.height);

  const det = await faceapi.detectSingleFace(video, options).withFaceLandmarks();
  if (det) {
    const { x:ux, y:uy, width:uw, height:uh } = det.detection.box;
    const { x:mx, y:my, width:mw, height:mh } = monaBox;
    ctx.drawImage(video, ux,uy,uw,uh, mx,my,mw,mh);
  }

  requestAnimationFrame(render);
}

// 사진 찍기 & 저장
snapBtn.onclick = () => {
  video.pause();
  video.srcObject.getTracks().forEach(t => t.stop());
};
saveBtn.onclick = () => {
  const a = document.createElement('a');
  a.download = 'swap.png';
  a.href     = canvas.toDataURL('image/png');
  a.click();
};
