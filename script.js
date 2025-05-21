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

  // --- 눈 사이 거리(비율 기반) ---
  const MONA_LEFT_EYE = 33, MONA_RIGHT_EYE = 263;
  const monaLeftEye = monaFaces[0].keypoints[MONA_LEFT_EYE];
  const monaRightEye = monaFaces[0].keypoints[MONA_RIGHT_EYE];
  const monaEyeDistPx = Math.hypot(
    monaLeftEye.x - monaRightEye.x,
    monaLeftEye.y - monaRightEye.y
  );
  const monaEyeDistNorm = monaEyeDistPx / monaWidth;

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

  // Mona Lisa 원본 mesh 생성 (중앙 기준, 0~1 범위 → 비디오 크기)
  const geometry  = new THREE.BufferGeometry();
  const positions = new Float32Array(monaPts.length * 3);
  const uvs       = new Float32Array(monaPts.length * 2);

  monaPts.forEach(([x,y,z], i) => {
    // Mona Lisa 좌표를 0~1로 정규화, 비디오 크기 기준으로 변환
    const normX = x / monaWidth;
    const normY = y / monaHeight;
    const px = (normX - 0.5) * vw;
    const py = (0.5 - normY) * vh;
    positions.set([px, py, 0], i*3);
    uvs.set([ normX, 1 - normY ], i*2);
  });

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv',       new THREE.BufferAttribute(uvs,       2));
  geometry.setIndex(indices);

  const texture = new THREE.TextureLoader().load('monalisa.jpg');
  const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  // 애니메이션 루프 (비율 동기화)
  async function animate() {
    const faces = await detector.estimateFaces(video);
    if (faces.length) {
      const kp  = faces[0].keypoints;
      // 내 눈사이 거리 (비율)
      const myLeftEye = kp[MONA_LEFT_EYE];
      const myRightEye = kp[MONA_RIGHT_EYE];
      const myEyeDistPx = Math.hypot(
        myLeftEye.x - myRightEye.x,
        myLeftEye.y - myRightEye.y
      );
      const myEyeDistNorm = myEyeDistPx / vw;
      // scale = 내 얼굴 눈사이거리(비율) / Mona Lisa 눈사이거리(비율)
      const scale = myEyeDistNorm / monaEyeDistNorm;
      mesh.scale.set(scale, scale, 1);

      // 얼굴 중심도 비율 기준
      const cxNorm = ((myLeftEye.x + myRightEye.x) / 2) / vw;
      const cyNorm = ((myLeftEye.y + myRightEye.y) / 2) / vh;
      mesh.position.x = (cxNorm - 0.5) * vw;
      mesh.position.y = (0.5 - cyNorm) * vh;
      mesh.position.z = 0;

      // 표정 변형도 동일 (비디오 기준 중앙좌표)
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
