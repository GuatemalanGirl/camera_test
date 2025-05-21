const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const captureBtn = document.getElementById('captureBtn');
const downloadLink = document.getElementById('downloadLink');

async function setup() {
  try {
    await faceapi.nets.tinyFaceDetector.loadFromUri('./models');
    await faceapi.nets.faceLandmark68Net.loadFromUri('./models');
    console.log("모델 로딩 완료");

    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (error) {
    alert('웹캠을 사용할 수 없거나 모델 로딩에 실패했습니다.');
    console.error(error);
  }
}

async function processFrame() {
  try {
    const detection = await faceapi.detectSingleFace(video, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks();
    if (!detection) {
      alert('얼굴을 인식하지 못했습니다. 다시 시도해 주세요.');
      return;
    }

    // 얼굴 영역 좌표
    const { x, y, width, height } = detection.detection.box;

    // 모나리자 이미지 로드
    const monalisaImg = new Image();
    monalisaImg.src = 'monalisa.jpg';

    monalisaImg.onload = () => {
      // 모나리자 배경 그리기
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(monalisaImg, 0, 0, canvas.width, canvas.height);

      // 사용자 얼굴 캡처
      ctx.drawImage(video, x, y, width, height, 240, 140, width, height); // 위치 조절 가능

      // 다운로드 링크 갱신
      downloadLink.href = canvas.toDataURL('image/png');
      downloadLink.style.display = 'inline';
    };
    monalisaImg.onerror = () => {
      alert('monalisa.jpg 이미지를 불러올 수 없습니다.');
    };
  } catch (err) {
    alert('얼굴 처리 중 오류가 발생했습니다.');
    console.error(err);
  }
}

captureBtn.addEventListener('click', processFrame);

// 실행
setup();
