//Import the THREE.js library
import * as THREE from "https://cdn.skypack.dev/three@0.129.0/build/three.module.js";
// To allow for the camera to move around the scene
import { OrbitControls } from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/controls/OrbitControls.js";
// To allow for importing the .gltf file
import { GLTFLoader } from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/loaders/GLTFLoader.js";

//Create a Three.JS Scene
//const scene = new THREE.Scene();
//create a new camera with positions and angles
//const container = document.getElementById('carousel');

//const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);

const modelPaths = [
    './gh-page/meshes/obj1/mesh.gltf',
    './gh-page/meshes/obj1/empty.gltf',
    './gh-page/meshes/obj1/mesh.gltf',
    './gh-page/meshes/obj1/mesh.gltf',
    './gh-page/meshes/obj1/empty.gltf',
    './gh-page/meshes/obj1/mesh.gltf'
];


//Keep track of the mouse position, so we can make the eye move
//let mouseX = container.clientWidth / 2;
//let mouseY = container.clientHeight / 2;

//Keep the 3D object on a global variable so we can access it later
let object;

//OrbitControls allow the camera to move around the scene
let controls;

//Set which object to render
let objToRender = 'eye';

//Instantiate a loader for the .gltf file
const loader = new GLTFLoader();

let currentIndex = 0;

const carousel = document.getElementById('carousel');
 // Function to create a 3D viewer inside a div
function createViewer(container, path) {
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);
    const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);
    console.log(container.clientHeight)
    // Lighting
    const light = new THREE.DirectionalLight(0xffffff, 1);
    light.position.set(5, 5, 5);
    scene.add(light);
    scene.add(new THREE.AmbientLight(0xffffff, 0.5));

    // Load model
    const loader = new GLTFLoader();
    let model;
    loader.load(path, function (gltf) {
        object = gltf.scene;
        object.scale.set(1, 1, 1);
        camera.position.set(0, 0.1, 1);
        scene.add(object);
        model = object
    },
    function (xhr) {
    //While it is loading, log the progress
    console.log((xhr.loaded / xhr.total * 100) + '% loaded');
    },
    function (error) {
    //If there is an error, log it
    console.error(error);
    });

    // Rotation controls
    let isDragging = false;
    let previousMouseX = 0;

    container.addEventListener("mousedown", (event) => {
        isDragging = true;
        previousMouseX = event.clientX;
    });

    window.addEventListener("mousemove", (event) => {
        if (isDragging && model) {
            let deltaX = event.clientX - previousMouseX;
            model.rotation.y += deltaX * 0.01; // Adjust rotation speed
            previousMouseX = event.clientX;
        }
    });

    window.addEventListener("mouseup", () => {
        isDragging = false;
    });

    // Animation loop
    function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
    }
    animate();

    // Handle resizing
    function onResize() {
        const width = container.clientWidth;
        const height = container.clientHeight;
        renderer.setSize(width, height);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        }
        window.addEventListener('resize', onResize);
    }

// Create a viewer for each model
modelPaths.forEach((path) => {
    const viewerDiv = document.createElement('div');
    viewerDiv.classList.add('viewer-container');
    carousel.appendChild(viewerDiv);
    createViewer(viewerDiv, path);
});

// Carousel navigation
document.getElementById('prev').addEventListener('click', () => {
    if (currentIndex > 0) {
        currentIndex--;
        carousel.style.transform = `translateX(-${currentIndex * 33}%)`;
    }
});

document.getElementById('next').addEventListener('click', () => {
    if (currentIndex + 2 < modelPaths.length - 1) {
        currentIndex++;
        carousel.style.transform = `translateX(-${currentIndex * 33}%)`;
    }
});

