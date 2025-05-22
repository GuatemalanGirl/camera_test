// script.js
import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.150.1/build/three.module.js';

(async function() {
  // 1) TF.js 백엔드 초기화 (WebGL → WASM 페일오버)
  try {
    await tf.setBackend('webgl');
    await tf.ready();
    console.log('TF.js backend:', tf.getBackend());
  } catch (e) {
    console.warn('WebGL backend failed, switching to WASM:', e);
    await tf.setBackend('wasm');
    await tf.ready();
    console.log('TF.js backend:', tf.getBackend());
  }

  // 2) DOM 요소 가져오기
  const video   = document.getElementById('video');
  const canvas  = document.getElementById('threejsCanvas');
  const monaImg = document.getElementById('mona');

  // 3) 카메라 스트림
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
  });
  video.srcObject = stream;
  await new Promise(r => video.onloadedmetadata = r);
  video.play();

  // 4) 캔버스 & 비디오 크기 동기화
  const vw = video.videoWidth, vh = video.videoHeight;
  video.width  = vw; video.height = vh;
  canvas.width = vw; canvas.height = vh;

  // Three.js 렌더러 초기화 (alpha: true로 투명 배경 허용)
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setSize(vw, vh);
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  // 씬과 카메라 (Orthographic)
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-vw/2, vw/2, vh/2, -vh/2, -1000, 1000);
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);

  function resizeCanvas() {
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    renderer.setSize(cw, ch);
    camera.left   = -cw/2;
    camera.right  =  cw/2;
    camera.top    =  ch/2;
    camera.bottom = -ch/2;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  // 5) FaceMesh detector 생성
  const detector = await faceLandmarksDetection.createDetector(
    faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
    { runtime: 'mediapipe', solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh', maxFaces: 1, refineLandmarks: true }
  );

  // 6) Mona Lisa 얼굴 포인트 계산
  await new Promise(r => monaImg.complete ? r() : monaImg.onload = r);
  const monaFaces = await detector.estimateFaces(monaImg);
  if (!monaFaces.length) throw new Error('Mona Lisa 얼굴을 인식할 수 없습니다.');
  const monaPts = monaFaces[0].keypoints.map(k => [k.x, k.y, k.z]);

  // 7) Delaunay 인덱스 생성
  const delaunay = Delaunator.from(monaPts.map(p => [p[0], p[1]]));
  const indices  = Array.from(delaunay.triangles);

  // 8) BufferGeometry + 초기 UV
  const geometry  = new THREE.BufferGeometry();
  const positions = new Float32Array(monaPts.length * 3);
  const uvs       = new Float32Array(monaPts.length * 2);

  monaPts.forEach(([x, y], i) => {
    const px = x - monaImg.naturalWidth/2;
    const py = (monaImg.naturalHeight - y) - monaImg.naturalHeight/2;
    positions.set([px, py, 0], i * 3);
    // 초기 UV는 그대로 두고 매 프레임 업데이트
    uvs.set([0, 0], i * 2);
  });

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv',       new THREE.BufferAttribute(uvs,       2));
  geometry.setIndex(indices);

  // 9) Mona Lisa 텍스처 로딩
  const loader = new THREE.TextureLoader();
  loader.setCrossOrigin('anonymous');
  const texture = loader.load('monalisa.jpg', () => {
    texture.generateMipmaps = false;
    texture.minFilter      = THREE.LinearFilter;
    texture.wrapS          = THREE.ClampToEdgeWrapping;
    texture.wrapT          = THREE.ClampToEdgeWrapping;
  });

  // 10) 비디오 텍스처 생성
  const videoTexture = new THREE.VideoTexture(video);
  videoTexture.minFilter = THREE.LinearFilter;
  videoTexture.magFilter = THREE.LinearFilter;
  videoTexture.wrapS     = THREE.ClampToEdgeWrapping;
  videoTexture.wrapT     = THREE.ClampToEdgeWrapping;
  videoTexture.flipY     = false;

  // 11) 배경용 Mona Lisa 원본 메쉬
  const bgGeometry = new THREE.PlaneGeometry(vw, vh);
  const bgMaterial = new THREE.MeshBasicMaterial({ map: texture });
  const bgMesh     = new THREE.Mesh(bgGeometry, bgMaterial);
  bgMesh.position.set(0, 0, -1);
  scene.add(bgMesh);

  // 12) 얼굴 합성 메쉬 (비디오)
  const faceMaterial = new THREE.MeshBasicMaterial({
    map: videoTexture,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.7
  });
  const faceMesh = new THREE.Mesh(geometry, faceMaterial);
  scene.add(faceMesh);

  // 13) 애니메이션 루프: 매 프레임 UV 업데이트
  async function animate() {
    const faces = await tf.tidy(() => detector.estimateFaces(video));
    if (faces.length) {
      const uvAttr = geometry.attributes.uv.array;
      faces[0].keypoints.forEach(({ x, y }, i) => {
        uvAttr[i*2]     = x / video.videoWidth;
        uvAttr[i*2 + 1] = 1 - (y / video.videoHeight);
      });
      geometry.attributes.uv.needsUpdate = true;
    }
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
  }
  animate();
})();
