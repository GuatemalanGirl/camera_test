import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.150.1/build/three.module.js';

(async function() {
  await tf.setBackend('webgl');
  await tf.ready();

  const video      = document.getElementById('video');
  const canvas     = document.getElementById('threejsCanvas');
  const monaImg    = document.getElementById('mona');

  // 1. 윈도우 리사이즈 함수 (video, canvas 등 참조 가능)
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

  // --- 눈 사이 거리(기준) ---
  const MONA_LEFT_EYE = 33, MONA_RIGHT_EYE = 263;
  const monaLeftEye = monaFaces[0].keypoints[MONA_LEFT_EYE];
  const monaRightEye = monaFaces[0].keypoints[MONA_RIGHT_EYE];
  const monaEyeDist = Math.hypot(
    monaLeftEye.x - monaRightEye.x,
    monaLeftEye.y - monaRightEye.y
  );

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

  // BufferGeometry + Material
  const geometry  = new THREE.BufferGeometry();
  const positions = new Float32Array(monaPts.length * 3);
  const uvs       = new Float32Array(monaPts.length * 2);

  monaPts.forEach(([x,y,z], i) => {
    const px = x - monaImg.naturalWidth/2;
    const py = (monaImg.naturalHeight - y) - monaImg.naturalHeight/2;
    positions.set([px, py, 0], i*3);
    uvs.set([ x / monaImg.naturalWidth, 1 - y / monaImg.naturalHeight ], i*2);
  });

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv',       new THREE.BufferAttribute(uvs,       2));
  geometry.setIndex(indices);

  const texture = new THREE.TextureLoader().load('monalisa.jpg');
  const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
  const mesh = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  // -----------------
  // 애니메이션 루프 (실시간 위치/크기 동기화)
  // -----------------
  async function animate() {
    const faces = await detector.estimateFaces(video);
    if (faces.length) {
      const kp  = faces[0].keypoints;

      // 내 얼굴의 눈 좌표 & 중심
      const myLeftEye = kp[MONA_LEFT_EYE];
      const myRightEye = kp[MONA_RIGHT_EYE];

      const myEyeDist = Math.hypot(
        myLeftEye.x - myRightEye.x,
        myLeftEye.y - myRightEye.y
      );

      // 내 얼굴 중심 (양 눈 중점)
      const cx = (myLeftEye.x + myRightEye.x) / 2;
      const cy = (myLeftEye.y + myRightEye.y) / 2;

      // Three.js 좌표 변환 (중앙 0,0)
      mesh.position.x = cx - vw / 2;
      mesh.position.y = vh / 2 - cy;
      mesh.position.z = 0;

      // 스케일
      const scale = myEyeDist / monaEyeDist;
      mesh.scale.set(scale, scale, 1);

      // 메쉬 변형(표정/형태)
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
