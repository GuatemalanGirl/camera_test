import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.150.1/build/three.module.js';

(async function() {
  await tf.setBackend('webgl');
  await tf.ready();

  const video   = document.getElementById('video');
  const canvas  = document.getElementById('threejsCanvas');
  const monaImg = document.getElementById('mona');

  function onWindowResize() {
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    video.width  = vw;  video.height = vh;
    canvas.width = vw;  canvas.height = vh;
    renderer.setSize(vw, vh);
    renderer.setPixelRatio(window.devicePixelRatio);
    camera.left   = -vw / 2;
    camera.right  =  vw / 2;
    camera.top    =  vh / 2;
    camera.bottom = -vh / 2;
    camera.updateProjectionMatrix();
  }

  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await new Promise(r => video.onloadedmetadata = r);
  video.play();

  const vw = video.videoWidth, vh = video.videoHeight;
  video.width  = vw; video.height = vh;
  canvas.width = vw; canvas.height = vh;

  const detector = await faceLandmarksDetection.createDetector(
    faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh, {
      runtime: 'mediapipe',
      solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
      maxFaces: 1,
      refineLandmarks: true
    }
  );

  await new Promise(r => monaImg.complete ? r() : monaImg.onload = r);
  const monaFaces = await detector.estimateFaces(monaImg);
  if (!monaFaces.length) throw new Error('Mona Lisa 얼굴을 인식할 수 없습니다.');
  const monaPts = monaFaces[0].keypoints.map(k => [k.x, k.y, k.z]);
  const monaWidth  = monaImg.naturalWidth;
  const monaHeight = monaImg.naturalHeight;

  // Helper: 눈 info
  function getFaceInfo(landmarks, idx1, idx2) {
    const a = landmarks[idx1], b = landmarks[idx2];
    const center = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    const dist = Math.hypot(a.x - b.x, a.y - b.y);
    const angle = Math.atan2(b.y - a.y, b.x - a.x);
    return { center, dist, angle };
  }

  // Mona Lisa 눈 info (기준)
  const MONA_LEFT_EYE = 33, MONA_RIGHT_EYE = 263;
  const monaInfo = getFaceInfo(monaFaces[0].keypoints, MONA_LEFT_EYE, MONA_RIGHT_EYE);

  // Delaunay 삼각분할
  const delaunay = Delaunator.from(monaPts.map(p => [p[0], p[1]]));
  const indices  = Array.from(delaunay.triangles);

  // Three.js 씬/카메라/렌더러
  const scene    = new THREE.Scene();
  const camera   = new THREE.OrthographicCamera(
    -vw/2, vw/2, vh/2, -vh/2, -1000, 1000
  );
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setSize(vw, vh);

  // 리사이즈 이벤트 등록
  onWindowResize();
  window.addEventListener('resize', onWindowResize);
  window.addEventListener('orientationchange', onWindowResize);

  // Mona Lisa Mesh 생성
  const geometry  = new THREE.BufferGeometry();
  const positions = new Float32Array(monaPts.length * 3);
  const uvs       = new Float32Array(monaPts.length * 2);

  monaPts.forEach(([x,y,z], i) => {
    // Mona Lisa 원본 이미지 중앙 기준, 3D 좌표계
    const px = x - monaWidth/2;
    const py = (monaHeight - y) - monaHeight/2;
    positions.set([px, py, 0], i*3);
    uvs.set([ x / monaWidth, 1 - y / monaHeight ], i*2);
  });

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv',       new THREE.BufferAttribute(uvs,       2));
  geometry.setIndex(indices);

  const texture = new THREE.TextureLoader().load('monalisa.jpg');
  const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  // 애니메이션 루프 (내 얼굴 기준 정렬)
  async function animate() {
    const faces = await detector.estimateFaces(video);
    if (faces.length) {
      const kp  = faces[0].keypoints;
      const myInfo = getFaceInfo(kp, MONA_LEFT_EYE, MONA_RIGHT_EYE);

      // (1) 스케일 적용
      const scale = myInfo.dist / monaInfo.dist;
      mesh.scale.set(scale, scale, 1);

      // (2) 회전 적용
      mesh.rotation.z = myInfo.angle - monaInfo.angle;

      // (3) 위치 이동 (Three.js 중앙좌표계로 변환)
      mesh.position.x = myInfo.center.x - vw/2;
      mesh.position.y = vh/2 - myInfo.center.y;
      mesh.position.z = 0;

      // (4) 표정/형태도 내 얼굴로 변형 (키포인트)
      const pos = geometry.attributes.position.array;
      kp.forEach(({x,y,z}, i) => {
        pos[i*3 + 0] = x - vw/2;
        pos[i*3 + 1] = vh/2 - y;
        pos[i*3 + 2] = 0;
      });
      geometry.attributes.position.needsUpdate = true;
    }

    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }
  animate();

})();
