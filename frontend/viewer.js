function load3DModel(meshUrl) {
    const container = document.getElementById("viewer-container");
    container.innerHTML = ""; // Clear previous model

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(0, 2, 5);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // ✅ Add Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1.5);
    directionalLight.position.set(1, 1, 1);
    scene.add(directionalLight);

    // ✅ Use OrbitControls Correctly
    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    // ✅ Load Model with GLTFLoader
    const loader = new THREE.GLTFLoader();
    loader.load(
        meshUrl,
        function (gltf) {
            const model = gltf.scene;
            scene.add(model);
            console.log("✅ Model Loaded:", model);
        },
        function (xhr) {
            console.log(`Loading: ${(xhr.loaded / xhr.total * 100).toFixed(2)}% complete`);
        },
        function (error) {
            console.error("❌ Error loading model:", error);
        }
    );

    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }

    animate();
}
