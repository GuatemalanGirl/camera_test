// script.js
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.150.1/build/three.module.js';

(async function() {
  // 1) TF.js WebGL 백엔드 초기화
  await tf.setBackend('webgl');
  await tf.ready();

  // 2) DOM 요소 참조
  const video      = document.getElementById('video');
  const canvas     = document.getElementById('threejsCanvas');
  const monaImg    = document.getElementById('mona');

  // 3) 윈도우 리사이즈 핸들러 정의 (이제 video, canvas 참조 가능)
  function onWindowResize() {
    const vw = window.innerWidth;
    const vh = window.innerHeight;

    // 1) 비디오 & 캔버스 크기 동기화
    video.width  = vw;  video.height = vh;
    canvas.width = vw;  canvas.height = vh;

    // 2) Three.js 렌더러 크기 업데이트
    renderer.setSize(vw, vh);
    renderer.setPixelRatio(window.devicePixelRatio);

    // 3) OrthographicCamera 프러스텀 재설정
    camera.left   = -vw / 2;
    camera.right  =  vw / 2;
    camera.top    =  vh / 2;
    camera.bottom = -vh / 2;
    camera.updateProjectionMatrix();
  }

  // 4) 웹캠 스트림 시작
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await new Promise(r => video.onloadedmetadata = r);
  video.play();

  // 5) 캔버스 & 비디오 크기 동기화
  const vw = video.videoWidth, vh = video.videoHeight;
  video.width  = vw; video.height = vh;
  canvas.width = vw; canvas.height = vh;

  // 6) FaceMesh detector 생성 (UMD 전역 객체 사용)
  const detector = await faceLandmarksDetection.createDetector(
    faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh, {
      runtime: 'mediapipe',
      solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
      maxFaces: 1,
      refineLandmarks: true
    }
  );

  // 7) Mona Lisa 이미지 3D 키포인트 한 번 추출
  await new Promise(r => monaImg.complete ? r() : monaImg.onload = r);
  const monaFaces = await detector.estimateFaces(monaImg);
  if (!monaFaces.length) throw new Error('Mona Lisa 얼굴을 인식할 수 없습니다.');
  const monaPts = monaFaces[0].keypoints.map(k => [k.x, k.y, k.z]);

  // 8) Delaunay 토폴로지 계산 (x,y 만 사용)
  const delaunay = Delaunator.from(monaPts.map(p => [p[0], p[1]]));
  const indices  = Array.from(delaunay.triangles);

  // 9) Three.js 씬 & OrthographicCamera 설정
  const scene    = new THREE.Scene();
  const camera   = new THREE.OrthographicCamera(
    -vw/2, vw/2,   // left, right
     vh/2, -vh/2,  // top, bottom
    -1000, 1000    // near, far
  );
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setSize(vw, vh);

  // 10) 리사이즈 이벤트 등록 (renderer, camera 정의 이후)
  onWindowResize();
  window.addEventListener('resize', onWindowResize);
  window.addEventListener('orientationchange', onWindowResize);

  // 11) BufferGeometry + Material 생성
  const geometry  = new THREE.BufferGeometry();
  const positions = new Float32Array(monaPts.length * 3);
  const uvs       = new Float32Array(monaPts.length * 2);

  monaPts.forEach(([x,y,z], i) => {
    // 이미지 좌표 → 중앙 기준 3D 좌표계
    const px = x - monaImg.naturalWidth/2;
    const py = (monaImg.naturalHeight - y) - monaImg.naturalHeight/2;
    // Orthographic이므로 z=0
    positions.set([px, py, 0], i*3);
    // UV 매핑
    uvs.set([ x / monaImg.naturalWidth, 1 - y / monaImg.naturalHeight ], i*2);
  });

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv',       new THREE.BufferAttribute(uvs,       2));
  geometry.setIndex(indices);

  const texture = new THREE.TextureLoader().load('monalisa.jpg');
  const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  // 12) 애니메이션 루프: 웹캠 키포인트로 메쉬 업데이트
  async function animate() {
    const faces = await detector.estimateFaces(video);
    if (faces.length) {
      const kp  = faces[0].keypoints;
      const pos = geometry.attributes.position.array;
      kp.forEach(({x,y,z}, i) => {
        // (x,y) 픽셀 → 중앙 기준 좌표계
        pos[i*3 + 0] = x - vw/2;
        pos[i*3 + 1] = vh/2 - y;
        // z는 0으로 고정해 평면 렌더링
        pos[i*3 + 2] = 0;
      });
      geometry.attributes.position.needsUpdate = true;
    }

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }
  animate();

})(); // end IIFE
