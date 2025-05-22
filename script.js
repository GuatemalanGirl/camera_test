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

  // 3) 카메라 스트림 (iOS 최적화: 전면 카메라, 640×480 이상 불필요)
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: 'user',
      width:  { ideal: 640 },
      height: { ideal: 480 }
    }
  });
  video.srcObject = stream;
  await new Promise(r => video.onloadedmetadata = r);
  video.play();

  // 4) 캔버스 & 비디오 크기 동기화
  const vw = video.videoWidth, vh = video.videoHeight;
  video.width  = vw; video.height = vh;
  canvas.width = vw; canvas.height = vh;

  // Three.js 렌더러 초기화 (antialias 유지)
  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setSize(vw, vh);
  renderer.setPixelRatio(window.devicePixelRatio || 1);

  // 씬과 카메라 (Orthographic)
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(
    -vw/2, vw/2,  // left, right
     vh/2, -vh/2, // top, bottom
    -1000, 1000   // near, far
  );
  camera.position.set(0, 0, 1);
  camera.lookAt(0, 0, 0);

  function resizeCanvas() {
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    // ① 렌더러 화면 크기 동기화
    renderer.setSize(cw, ch);
    // ② Orthographic 카메라 프로젝션 동기화
    camera.left   = -cw/2;
    camera.right  =  cw/2;
    camera.top    =  ch/2;
    camera.bottom = -ch/2;
    camera.updateProjectionMatrix();
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();  // 최초 한 번 호출


  // 5) FaceMesh detector 생성
  const detector = await faceLandmarksDetection.createDetector(
    faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh, {
      runtime: 'mediapipe',
      solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh',
      maxFaces: 1,
      refineLandmarks: true
    }
  );

  // 6) Mona Lisa 이미지 한 번만 분석해서 키포인트 저장
  await new Promise(r => monaImg.complete ? r() : monaImg.onload = r);
  const monaFaces = await detector.estimateFaces(monaImg);
  if (!monaFaces.length) throw new Error('Mona Lisa 얼굴을 인식할 수 없습니다.');
  const monaPts = monaFaces[0].keypoints.map(k => [k.x, k.y, k.z]);

  // 7) Delaunay 트라이앵글 인덱스 생성
  const delaunay = Delaunator.from(monaPts.map(p => [p[0], p[1]]));
  const indices  = Array.from(delaunay.triangles);

  // 8) BufferGeometry + UV 매핑
  const geometry  = new THREE.BufferGeometry();
  const positions = new Float32Array(monaPts.length * 3);
  const uvs       = new Float32Array(monaPts.length * 2);

  monaPts.forEach(([x, y], i) => {
    // 이미지 좌표 → 중앙 기준 3D 좌표계
    const px =  x - monaImg.naturalWidth  / 2;
    const py = (monaImg.naturalHeight - y) - monaImg.naturalHeight / 2;
    positions.set([px, py, 0], i * 3);
    // UV 매핑
    uvs.set([ x / monaImg.naturalWidth, 1 - y / monaImg.naturalHeight ], i * 2);
  });

  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('uv',       new THREE.BufferAttribute(uvs,       2));
  geometry.setIndex(indices);

  // 9) Texture 로딩 (crossOrigin + NPOT WebGL 안전)
  const loader = new THREE.TextureLoader();
  loader.setCrossOrigin('anonymous');
  const texture = loader.load('monalisa.jpg', () => {
    texture.generateMipmaps = false;
    texture.minFilter      = THREE.LinearFilter;
    texture.wrapS          = THREE.ClampToEdgeWrapping;
    texture.wrapT          = THREE.ClampToEdgeWrapping;
  });

  const material = new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide });
  const mesh     = new THREE.Mesh(geometry, material);
  scene.add(mesh);

  // 10) 애니메이션 루프: 매 프레임 FaceMesh → 메모리 릭 방지(tf.tidy)
  async function animate() {
  const faces = await tf.tidy(() => detector.estimateFaces(video));
  if (faces.length) {
    const { clientWidth: cw, clientHeight: ch } = canvas;
    const pos = geometry.attributes.position.array;
    faces[0].keypoints.forEach(({ x, y }, i) => {
      // 1) 비디오 픽셀 좌표 → CSS 픽셀 좌표
      const sx = ( x / video.videoWidth  ) * cw;
      const sy = ( y / video.videoHeight ) * ch;
      // 2) CSS 픽셀 좌표 → Three.js 월드좌표
      pos[i*3 + 0] = sx - cw/2;
      pos[i*3 + 1] = ch/2 - sy;
      pos[i*3 + 2] = 0;
    });
    geometry.attributes.position.needsUpdate = true;
  }
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();

})();
