//Import the THREE.js library
import * as THREE from "https://cdn.skypack.dev/three@0.129.0/build/three.module.js";
// To allow for the camera to move around the scene
import { OrbitControls } from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/controls/OrbitControls.js";
// To allow for importing the .gltf file
import { GLTFLoader } from "https://cdn.skypack.dev/three@0.129.0/examples/jsm/loaders/GLTFLoader.js";

//Create a Three.JS Scene
const scene = new THREE.Scene();
//create a new camera with positions and angles
const container = document.getElementById('container3D');

const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.1, 1000);

const modelPaths = [
    './gh-page/meshes/obj1/mesh.gltf',
    './gh-page/meshes/obj1/empty.gltf',
    './gh-page/meshes/obj1/mesh.gltf',
    './gh-page/meshes/obj1/mesh.gltf',
    './gh-page/meshes/obj1/empty.gltf',
    './gh-page/meshes/obj1/mesh.gltf'
];

console.log(container.clientHeight)
//Keep track of the mouse position, so we can make the eye move
let mouseX = container.clientWidth / 2;
let mouseY = container.clientHeight / 2;

//Keep the 3D object on a global variable so we can access it later
let object;

//OrbitControls allow the camera to move around the scene
let controls;

//Set which object to render
let objToRender = 'eye';

//Instantiate a loader for the .gltf file
const loader = new GLTFLoader();
let models = [];
let currentIndex = 0;
const dist = 7;
//Load the file
modelPaths.forEach((path, index) => {
            loader.load(path, function (gltf) {
    //If the file is loaded, add it to the scene
    object = gltf.scene;
//    object.scale.set(1, 1, 1)
    object.position.set(
                    index/modelPaths.length * dist - 0.5*dist,
                    0,
                    0,
//                    0
                );
    scene.add(object);
    models.push(object);
  },
  function (xhr) {
    //While it is loading, log the progress
    console.log((xhr.loaded / xhr.total * 100) + '% loaded');
  },
  function (error) {
    //If there is an error, log it
    console.error(error);
  }
    );
});


//Instantiate a new renderer and set its size
const renderer = new THREE.WebGLRenderer({ alpha: true }); //Alpha: true allows for the transparent background
renderer.setSize(container.clientWidth, container.clientHeight);

//Add the renderer to the DOM
document.getElementById("container3D").appendChild(renderer.domElement);

//Set how far the camera will be from the 3D model
//camera.position.z = objToRender === "dino" ? 25 : 500;
camera.position.set(0, 0.1, 1);
//Add lights to the scene, so we can actually see the 3D model
const topLight = new THREE.DirectionalLight(0xffffff, 1); // (color, intensity)
topLight.position.set(0, 0.2, 1) //top-left-ish
topLight.castShadow = true;
scene.add(topLight);

const ambientLight = new THREE.AmbientLight(0x333333, objToRender === "dino" ? 5 : 1);
scene.add(ambientLight);

//This adds controls to the camera, so we can rotate / zoom it with the mouse
if (objToRender === "dino") {
  controls = new OrbitControls(camera, renderer.domElement);
}

// Rotation flags
let isMouseDown = false;

// Mouse down event to start rotating
window.addEventListener('mousedown', () => {
    isMouseDown = true;
});

// Mouse up event to stop rotating
window.addEventListener('mouseup', () => {
    isMouseDown = false;
});

//Render the scene
function animate() {
  requestAnimationFrame(animate);
  //Here we could add some code to update the scene, adding some automatic movement

  //Make the eye move
  if (isMouseDown && objToRender === "eye") {
    //I've played with the constants here until it looked good
    modelPaths[modelPaths.length/2].rotation.y = -3 + mouseX / container.clientWidth * 3;
//    object.rotation.x = -1.2 + mouseY * 0.5 / window.innerHeight;
  }
  renderer.render(scene, camera);
}

//Add a listener to the window, so we can resize the window and the camera
window.addEventListener("resize", function () {
  camera.aspect = 1
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});

//add mouse position listener, so we can make the eye move
document.onmousemove = (e) => {
  mouseX = e.clientX;
  mouseY = e.clientY;
}

//Start the 3D rendering
animate();

function updateCarousel() {
//    const angle = (currentIndex / modelPaths.length) * Math.PI * 2;
    models.forEach((model, index) => {
//        const newAngle = angle + (index / modelPaths.length) * Math.PI * 2;
        model.position.set(
//            (currentIndex + index)/modelPaths.length * dist - 0.5*dist,
            ((currentIndex + index)%modelPaths.length)/modelPaths.length  * dist - 0.5*dist,
            0,
            0,
//            radius * Math.sin(newAngle),
//            0
        );
    });
}

// Button click handlers
document.getElementById('prev').addEventListener('click', () => {
    currentIndex = (currentIndex - 1 + modelPaths.length) % modelPaths.length;
    updateCarousel();
});

document.getElementById('next').addEventListener('click', () => {
    currentIndex = (currentIndex + 1) % modelPaths.length;
    updateCarousel();
});
// Handle resizing
function onResize() {
    const width = container.clientWidth;
    const height = container.clientHeight;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
}
window.addEventListener('resize', onResize);
