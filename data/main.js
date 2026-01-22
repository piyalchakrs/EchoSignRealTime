// ========== 3D HAND MODEL VISUALIZATION ==========
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

// Hand model parameters - initial straight position
const HAND_PARAMS = {
    wrist: 0,
    thumb: 0,
    index: 0,
    middle: 0,
    ring: 0,
    pinky: 0,
    thumbz: -0.15,
    indexz: -0.3,
    middlez: -0.08,
    ringz: 0.22,
    pinkyz: 0.52
};

// Calibration offsets - adjust these based on your sensor baseline values
const FLEX_OFFSETS = {
    thumb: 0.20,   // Increased to make thumb straighter at rest
    index: 0.15,
    middle: 0.15,
    ring: 0.15,
    pinky: 0.15
};

let scene, camera, renderer, orbitControls;
let hand3DModel = null;
let handRotationArrays = null;  // Will store rotation objects for GSAP
let animationFrameId;

// Smoothing for gyro data to prevent jitter
let smoothedRotation = { x: 0, y: 0, z: 0 };
let smoothedFingers = {
    thumb: 0,
    index: 0,
    middle: 0,
    ring: 0,
    pinky: 0
};
const SMOOTHING_FACTOR = 0.3;  // Balanced smoothing
const FINGER_SMOOTHING = 0.2;  // Finger smoothing

// Frame limiting to prevent excessive updates
let lastUpdateTime = 0;
const UPDATE_INTERVAL = 16;  // ~60 FPS

// Initialize 3D Hand Model
function init3DHandModel() {
    const container = document.getElementById('hand3d');
    if (!container) {
        console.error('3D container not found');
        return;
    }

    // Create scene with clean background
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1e1b4b);  // Deep purple background

    // Create camera
    const width = container.clientWidth || 800;
    const height = container.clientHeight || 500;
    console.log('Container size:', width, 'x', height);
    
    camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 100);
    camera.position.set(0, 0, 5);
    scene.add(camera);

    // Create renderer with high quality settings
    renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true,
        powerPreference: 'high-performance'
    });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);  // Use full device pixel ratio
    renderer.shadowMap.enabled = false;  // Disabled for better performance
    renderer.outputEncoding = THREE.sRGBEncoding;  // Better color accuracy
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    container.appendChild(renderer.domElement);
    console.log('Renderer canvas added:', renderer.domElement);

    // Add orbit controls
    orbitControls = new OrbitControls(camera, renderer.domElement);
    orbitControls.target.set(0, 0, 0);
    orbitControls.enableDamping = true;
    orbitControls.maxPolarAngle = Math.PI / 2;
    orbitControls.minDistance = 3;
    orbitControls.maxDistance = 10;

    // Enhanced lighting setup for richer appearance
    const ambientLight = new THREE.AmbientLight(0xf5f5f5, 0.6);
    scene.add(ambientLight);

    const mainLight = new THREE.DirectionalLight(0xffffff, 1.2);
    mainLight.position.set(5, 8, 5);
    scene.add(mainLight);

    const fillLight = new THREE.DirectionalLight(0xe6f0ff, 0.5);
    fillLight.position.set(-5, 3, -3);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xffffff, 0.4);
    rimLight.position.set(0, 5, -6);
    scene.add(rimLight);

    // Add hemisphere light for better ambient feel
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444444, 0.3);
    scene.add(hemiLight);

    // Load GLTF hand model
    const gltfLoader = new GLTFLoader();
    console.log('Starting to load hand.glb model from:', window.location.origin + '/hand.glb');
    
    gltfLoader.load(
        '/hand.glb',  // Use absolute path from root
        (gltf) => {
            console.log('✓ GLTF loaded successfully!');
            hand3DModel = gltf.scene;
            
            // Keep the hand as-is (model is already right-handed)
            hand3DModel.scale.set(1, 1, 1);
            
            // Apply materials
            hand3DModel.traverse((child) => {
                console.log('Child:', child.name, child.type);
                if (child.isMesh) {
                    // Apply high-quality skin material
                    child.material = new THREE.MeshStandardMaterial({
                        color: 0xFFCBA4,
                        roughness: 0.75,
                        metalness: 0.0,
                        side: THREE.DoubleSide,
                        flatShading: false
                    });
                    // Enable smooth shading
                    child.geometry.computeVertexNormals();
                    console.log('Applied material to mesh:', child.name);
                }
            });
            
            scene.add(hand3DModel);
            console.log('✓ Hand model added to scene');
            
            // Setup bones for animation (this sets rotation and initial pose)
            setupHandBones();
        },
        (progress) => {
            const percent = progress.total > 0 ? (progress.loaded / progress.total * 100).toFixed(0) : '?';
            console.log(`Loading hand model: ${percent}%`);
        },
        (error) => {
            console.error('✗ ERROR loading hand model:', error);
            console.log('Error details:', error.message);
            console.log('Make sure hand.glb file exists in the data folder!');
            // Show error message to user
            const container = document.getElementById('hand3d');
            if (container) {
                container.innerHTML = '<div style="color: #ff6b6b; padding: 20px; text-align: center;">Failed to load 3D hand model.<br>Check console for details.</div>';
            }
        }
    );

    // Handle window resize
    window.addEventListener('resize', onWindowResize, false);

    // Reset camera button
    document.getElementById('resetCamera')?.addEventListener('click', resetCamera);

    // Start animation loop
    animate3DScene();

    console.log('3D Hand Model initialized');
}

// Setup hand bones for animation (matching Gill003's setBones function exactly)
function setupHandBones() {
    const hand = scene.getObjectByName('Hand');
    if (!hand || !hand.skeleton) {
        console.error('Hand mesh or skeleton not found!');
        return;
    }
    
    console.log('Setting up hand bones...');
    
    // Rotate hand parent (this is the key!)
    hand.parent.rotation.x = Math.PI / 2;  // 90 degrees
    hand.parent.rotation.y = 0;
    hand.parent.rotation.z = Math.PI;      // 180 degrees
    
    const bones = hand.skeleton.bones;
    console.log('Total bones:', bones.length);
    
    // Wrist bones
    const wrist = bones[0];
    const wrist1 = bones[1];
    const wrist2 = bones[2];
    const wrist3 = bones[6];
    const wrist4 = bones[10];
    const wrist5 = bones[14];
    const wrist6 = bones[18];
    
    wrist1.scale.set(1, 1.1, 1);
    wrist1.rotation.x = HAND_PARAMS.wrist;
    wrist2.rotation.x = HAND_PARAMS.wrist;
    wrist3.rotation.x = HAND_PARAMS.wrist;
    wrist4.rotation.x = HAND_PARAMS.wrist;
    wrist5.rotation.x = HAND_PARAMS.wrist;
    wrist6.rotation.x = HAND_PARAMS.wrist;
    
    // Thumb bones
    const thumb1 = bones[3];
    const thumb2 = bones[4];
    const thumb3 = bones[5];
    thumb1.scale.set(0.9, 1.3, 0.9);
    thumb1.rotation.x = HAND_PARAMS.thumb;
    thumb2.rotation.x = HAND_PARAMS.thumb;
    thumb3.rotation.x = HAND_PARAMS.thumb;
    thumb1.rotation.z = HAND_PARAMS.thumbz;
    thumb2.rotation.z = HAND_PARAMS.thumbz;
    thumb3.rotation.z = HAND_PARAMS.thumbz;
    
    // Index bones
    const index1 = bones[7];
    const index2 = bones[8];
    const index3 = bones[9];
    index1.scale.set(0.9, 1.3, 0.9);
    index1.rotation.x = HAND_PARAMS.index;
    index2.rotation.x = HAND_PARAMS.index;
    index3.rotation.x = HAND_PARAMS.index;
    
    // Middle bones
    const middle1 = bones[11];
    const middle2 = bones[12];
    const middle3 = bones[13];
    middle1.scale.set(0.9, 1.3, 0.9);
    middle1.rotation.x = HAND_PARAMS.middle;
    middle2.rotation.x = HAND_PARAMS.middle;
    middle3.rotation.x = HAND_PARAMS.middle;
    
    // Ring bones
    const ring1 = bones[15];
    const ring2 = bones[16];
    const ring3 = bones[17];
    ring1.scale.set(0.9, 1.3, 0.9);
    ring1.rotation.x = HAND_PARAMS.ring;
    ring2.rotation.x = HAND_PARAMS.ring;
    ring3.rotation.x = HAND_PARAMS.ring;
    
    // Pinky bones
    const pinky1 = bones[19];
    const pinky2 = bones[20];
    const pinky3 = bones[21];
    pinky1.scale.set(0.9, 1.3, 0.9);
    pinky1.rotation.x = HAND_PARAMS.pinky;
    pinky2.rotation.x = HAND_PARAMS.pinky;
    pinky3.rotation.x = HAND_PARAMS.pinky;
    
    // Store rotation objects for GSAP animation
    handRotationArrays = {
        hand: hand.parent.rotation,
        wrist: [wrist.rotation, wrist1.rotation, wrist2.rotation, wrist3.rotation, wrist4.rotation, wrist5.rotation, wrist6.rotation],
        thumb: [thumb1.rotation, thumb2.rotation, thumb3.rotation],
        index: [index1.rotation, index2.rotation, index3.rotation],
        middle: [middle1.rotation, middle2.rotation, middle3.rotation],
        ring: [ring1.rotation, ring2.rotation, ring3.rotation],
        pinky: [pinky1.rotation, pinky2.rotation, pinky3.rotation]
    };
    
    console.log('Hand bones setup complete!');
}

// Create simple hand mesh using basic geometries
function createSimpleHandMesh() {
    const handGroup = new THREE.Group();
    handGroup.name = 'SimpleHand';

    const material = new THREE.MeshStandardMaterial({
        color: 0xffdbac,
        roughness: 0.7,
        metalness: 0.1
    });

    // Palm
    const palmGeometry = new THREE.BoxGeometry(0.8, 0.15, 1.0);
    const palm = new THREE.Mesh(palmGeometry, material);
    palm.castShadow = true;
    palm.receiveShadow = true;
    handGroup.add(palm);

    // Finger configurations: [xOffset, zOffset, length, numSegments]
    const fingerConfigs = [
        { name: 'thumb', x: -0.35, z: 0.3, length: 0.5, segments: 2, baseAngle: 45 },
        { name: 'index', x: -0.25, z: -0.5, length: 0.7, segments: 3, baseAngle: 0 },
        { name: 'middle', x: 0, z: -0.55, length: 0.75, segments: 3, baseAngle: 0 },
        { name: 'ring', x: 0.25, z: -0.5, length: 0.65, segments: 3, baseAngle: 0 },
        { name: 'pinky', x: 0.35, z: -0.4, length: 0.55, segments: 3, baseAngle: 0 }
    ];

    fingerConfigs.forEach(config => {
        const fingerRoot = new THREE.Group();
        fingerRoot.position.set(config.x, 0.075, config.z);
        
        // Apply base angle for thumb
        if (config.baseAngle !== 0) {
            fingerRoot.rotation.y = THREE.MathUtils.degToRad(config.baseAngle);
        }
        
        fingerBones[config.name] = [];
        const segmentLength = config.length / config.segments;
        
        let currentParent = fingerRoot;
        
        for (let i = 0; i < config.segments; i++) {
            const segmentGroup = new THREE.Group();
            segmentGroup.position.z = segmentLength; // Move forward from parent
            
            const segmentGeometry = new THREE.CylinderGeometry(0.06, 0.055, segmentLength, 8);
            const segmentMesh = new THREE.Mesh(segmentGeometry, material);
            segmentMesh.rotation.x = Math.PI / 2;
            segmentMesh.position.z = segmentLength / 2;
            segmentMesh.castShadow = true;
            segmentMesh.name = `${config.name}_segment_${i}`;
            
            segmentGroup.add(segmentMesh);
            currentParent.add(segmentGroup);
            
            fingerBones[config.name].push(segmentGroup);
            currentParent = segmentGroup;
        }
        
        handGroup.add(fingerRoot);
    });

    handGroup.position.set(0, 0.5, 0);
    handGroup.rotation.y = Math.PI;
    scene.add(handGroup);
    hand3DModel = handGroup;

    console.log('Simple hand mesh created');
}

// Animation loop
function animate3DScene() {
    animationFrameId = requestAnimationFrame(animate3DScene);

    if (orbitControls) {
        orbitControls.update();
    }

    if (renderer && scene && camera) {
        renderer.render(scene, camera);
    }
}

// Reset camera position
function resetCamera() {
    if (camera && orbitControls) {
        camera.position.set(0, 1, 3);
        orbitControls.target.set(0, 0.5, 0);
        orbitControls.update();
    }
}

// Handle window resize
function onWindowResize() {
    const container = document.getElementById('hand3d');
    if (!container) return;

    const width = container.clientWidth;
    const height = container.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

// Update hand pose based on sensor data
function update3DHandPose(data) {
    if (!handRotationArrays) {
        // Not ready yet
        return;
    }

    // Frame limiting - only update every UPDATE_INTERVAL ms
    const now = Date.now();
    if (now - lastUpdateTime < UPDATE_INTERVAL) {
        return;
    }
    lastUpdateTime = now;

    // Apply calibration offsets and map flex sensor values to rotation angles
    // Subtract offset so that baseline = 0 (straight), then map to curl angle
    const thumbRaw = Math.max(0, (data.f1 || 0) - FLEX_OFFSETS.thumb);
    const indexRaw = Math.max(0, (data.f2 || 0) - FLEX_OFFSETS.index);
    const middleRaw = Math.max(0, (data.f3 || 0) - FLEX_OFFSETS.middle);
    const ringRaw = Math.max(0, (data.f4 || 0) - FLEX_OFFSETS.ring);
    const pinkyRaw = Math.max(0, (data.f5 || 0) - FLEX_OFFSETS.pinky);
    
    // Convert to rotation angles with maximum sensitivity for full curling
    const targetThumb = -thumbRaw * 5.0;   // Increased sensitivity
    const targetIndex = -indexRaw * 6.0;   // Increased sensitivity
    const targetMiddle = -middleRaw * 5.5; // Increased sensitivity
    const targetRing = -ringRaw * 6.0;     // Increased sensitivity
    const targetPinky = -pinkyRaw * 6.5;   // Maximum sensitivity
    
    // Smooth finger rotations with increased response
    smoothedFingers.thumb += (targetThumb - smoothedFingers.thumb) * 0.35;
    smoothedFingers.index += (targetIndex - smoothedFingers.index) * 0.35;
    smoothedFingers.middle += (targetMiddle - smoothedFingers.middle) * 0.35;
    smoothedFingers.ring += (targetRing - smoothedFingers.ring) * 0.35;
    smoothedFingers.pinky += (targetPinky - smoothedFingers.pinky) * 0.35;
    
    // Map gyro values to hand rotation with correct axis mapping
    // Apply smoothing to reduce jitter
    const targetRotX = (data.gx || 0) * 0.015;   // Pitch (palm up/down) - gyroX
    const targetRotY = (data.gy || 0) * 0.015;   // Yaw (rotate left/right) - gyroY (removed inversion)
    const targetRotZ = (data.gz || 0) * 0.015;   // Roll (twist wrist) - gyroZ

    // Smooth the rotation using exponential moving average
    smoothedRotation.x += (targetRotX - smoothedRotation.x) * SMOOTHING_FACTOR;
    smoothedRotation.y += (targetRotY - smoothedRotation.y) * SMOOTHING_FACTOR;
    smoothedRotation.z += (targetRotZ - smoothedRotation.z) * SMOOTHING_FACTOR;

    // Apply smoothed rotation directly (no GSAP to prevent reverting)
    handRotationArrays.hand.x = smoothedRotation.x;
    handRotationArrays.hand.y = smoothedRotation.y;
    handRotationArrays.hand.z = smoothedRotation.z;

    // Apply smoothed rotations directly
    // Wrist bones - keep neutral
    handRotationArrays.wrist.forEach(bone => bone.x = 0);
    
    // Thumb
    handRotationArrays.thumb.forEach(bone => {
        bone.x = smoothedFingers.thumb;
        bone.z = -0.05;
    });
    
    // Index
    handRotationArrays.index.forEach(bone => bone.x = smoothedFingers.index);
    handRotationArrays.index[0].z = -0.03;
    
    // Middle
    handRotationArrays.middle.forEach(bone => bone.x = smoothedFingers.middle);
    handRotationArrays.middle[0].z = 0;
    
    // Ring
    handRotationArrays.ring.forEach(bone => bone.x = smoothedFingers.ring);
    handRotationArrays.ring[0].z = 0.03;
    
    // Pinky
    handRotationArrays.pinky.forEach(bone => bone.x = smoothedFingers.pinky);
    handRotationArrays.pinky[0].z = 0.05;
}

// ========== END 3D HAND MODEL ==========

// WebSocket connection for real-time data
let ws = null;
let reconnectTimeout = null;
let sampleCount = 0;
let predCount = 0;
let predRateInterval = null;
let lastPredTime = Date.now();
let historyItems = [];
const MAX_HISTORY = 20;
let voiceEnabled = false;
let lastSpoken = '';
let speakCooldownMs = 1500; // minimal delay between speeches
let lastSpeakTime = 0;

function speakText(text) {
    try {
        if (!voiceEnabled) return;
        if (!text || text === 'unknown') return;
        const now = Date.now();
        if (text === lastSpoken && (now - lastSpeakTime) < speakCooldownMs) return;
        // Use Web Speech API
        const utter = new SpeechSynthesisUtterance(text);
        // Prefer a neutral English voice if available
        const voices = window.speechSynthesis.getVoices();
        const enVoice = voices.find(v => /en/i.test(v.lang));
        if (enVoice) utter.voice = enVoice;
        utter.rate = 1.0;
        utter.pitch = 1.0;
        window.speechSynthesis.cancel(); // stop any ongoing speech to avoid overlap
        window.speechSynthesis.speak(utter);
        lastSpoken = text;
        lastSpeakTime = now;
    } catch (e) {
        console.warn('Speech synthesis error:', e);
    }
}

// DOM elements
const elements = {
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    gestureName: document.getElementById('gestureName'),
    confidenceFill: document.getElementById('confidenceFill'),
    confidenceBar: null, // Will be set after DOM loads
    confidenceValue: document.getElementById('confidenceValue'),
    gdpValue: document.getElementById('gdpValue'),
    gdpBar: document.getElementById('gdpBar'),
    sampleCount: document.getElementById('sampleCount'),
    predRate: document.getElementById('predRate'),
    thumbBar: document.getElementById('thumbBar'),
    thumbVal: document.getElementById('thumbVal'),
    indexBar: document.getElementById('indexBar'),
    indexVal: document.getElementById('indexVal'),
    middleBar: document.getElementById('middleBar'),
    middleVal: document.getElementById('middleVal'),
    ringBar: document.getElementById('ringBar'),
    ringVal: document.getElementById('ringVal'),
    pinkyBar: document.getElementById('pinkyBar'),
    pinkyVal: document.getElementById('pinkyVal'),
    accelX: document.getElementById('accelX'),
    accelY: document.getElementById('accelY'),
    accelZ: document.getElementById('accelZ'),
    gyroX: document.getElementById('gyroX'),
    gyroY: document.getElementById('gyroY'),
    gyroZ: document.getElementById('gyroZ'),
    historyList: document.getElementById('historyList'),
    debugLog: document.getElementById('debugLog')
};

// Log management
let logCounter = 0;
const MAX_LOG_ENTRIES = 50;

function addLog(message, type = 'gesture') {
    if (!elements.debugLog) return;
    
    logCounter++;
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    elements.debugLog.appendChild(entry);
    
    // Keep only last MAX_LOG_ENTRIES
    while (elements.debugLog.children.length > MAX_LOG_ENTRIES) {
        elements.debugLog.removeChild(elements.debugLog.firstChild);
    }
    
    // Auto-scroll to bottom
    elements.debugLog.scrollTop = elements.debugLog.scrollHeight;
}

// Initialize WebSocket connection
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    try {
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            console.log('WebSocket connected');
            updateStatus('connected', 'Connected');
            clearTimeout(reconnectTimeout);
        };
        
        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleData(data);
            } catch (e) {
                console.error('Failed to parse message:', e);
            }
        };
        
        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateStatus('error', 'Connection Error');
        };
        
        ws.onclose = () => {
            console.log('WebSocket disconnected');
            updateStatus('disconnected', 'Disconnected');
            reconnectTimeout = setTimeout(connectWebSocket, 3000);
        };
    } catch (e) {
        console.error('Failed to create WebSocket:', e);
        updateStatus('error', 'Connection Failed');
        reconnectTimeout = setTimeout(connectWebSocket, 5000);
    }
}

// Update connection status
function updateStatus(status, text) {
    elements.statusDot.className = `status-dot ${status}`;
    elements.statusText.textContent = text;
}

// Handle incoming data
function handleData(data) {
    // Mark as connected when receiving data
    if (elements.statusDot.className !== 'status-dot connected') {
        updateStatus('connected', 'Connected');
    }
    
    sampleCount++;
    elements.sampleCount.textContent = sampleCount.toLocaleString();
    
    // Update 3D hand model with real-time sensor data
    update3DHandPose(data);
    
    // Check if we're in sentence mode (UI state takes priority)
    if (currentMode === 'sentence') {
        // In sentence mode, only show sentence predictions and progress
        if (data.mode === 'sentence') {
            if (data.recording === true) {
                // Show prediction progress
                elements.gestureName.textContent = '⏳ Analyzing...';
                const progress = (data.progress || 0) * 100;
                updateConfidence(progress);
                
                // Update progress bar style for sentence mode
                elements.confidenceBar.style.background = 'linear-gradient(to right, #10b981, #34d399)';
            } else if (data.sentence) {
                // Show sentence prediction
                elements.gestureName.textContent = data.sentence;
                predCount++;
                addToHistory(data.sentence);
                if (typeof window.__lastSentenceSpoken === 'undefined') {
                    window.__lastSentenceSpoken = null;
                }
                if (data.sentence !== window.__lastSentenceSpoken) {
                    speakText(data.sentence);
                    window.__lastSentenceSpoken = data.sentence;
                }
                
                const confidence = data.confidence ? data.confidence * 100 : 0;
                updateConfidence(confidence);
                
                // Reset progress bar style
                elements.confidenceBar.style.background = '';
                
                console.log(`Predicted: "${data.sentence}" (conf: ${confidence.toFixed(0)}%, meanD: ${data.meanD?.toFixed(1)})`);
            }
        } else {
            // ESP32 is sending gesture data but we're in sentence mode
            // Show waiting message until next sentence prediction triggers
            if (!elements.gestureName.textContent.includes('...')) {
                elements.gestureName.textContent = '⏳ Waiting...';
                updateConfidence(0);
            }
        }
        // Don't return - continue to update sensor data below
    }
    
    // Gesture mode - update gesture display
    if (currentMode === 'gesture') {
        // Update gesture display
        if (data.label && data.label !== 'unknown') {
            elements.gestureName.textContent = data.label;
            predCount++;
            addToHistory(data.label);
            if (typeof window.__lastGestureSpoken === 'undefined') {
                window.__lastGestureSpoken = null;
            }
            if (data.label !== window.__lastGestureSpoken) {
                speakText(data.label);
                window.__lastGestureSpoken = data.label;
            }
            
            // Calculate confidence (inverse of meanD, normalized)
            const confidence = data.meanD ? Math.max(0, Math.min(100, 100 - data.meanD * 10)) : 0;
            updateConfidence(confidence);
        }
    }
    
    // Update sensor data (GDP, flex, IMU) - for both modes
    if (data.gdp !== undefined) {
        const gdp = parseFloat(data.gdp);
        elements.gdpValue.textContent = gdp.toFixed(1);
        const gdpPercent = Math.min(100, (gdp / 50) * 100);
        elements.gdpBar.style.width = `${gdpPercent}%`;
    }
    
    // Update flex sensors
    updateFlexSensor('thumb', data.f1);
    updateFlexSensor('index', data.f2);
    updateFlexSensor('middle', data.f3);
    updateFlexSensor('ring', data.f4);
    updateFlexSensor('pinky', data.f5);
    
    // Update IMU data
    if (data.ax !== undefined) elements.accelX.textContent = parseFloat(data.ax).toFixed(2);
    if (data.ay !== undefined) elements.accelY.textContent = parseFloat(data.ay).toFixed(2);
    if (data.az !== undefined) elements.accelZ.textContent = parseFloat(data.az).toFixed(2);
    if (data.gx !== undefined) elements.gyroX.textContent = parseFloat(data.gx).toFixed(1);
    if (data.gy !== undefined) elements.gyroY.textContent = parseFloat(data.gy).toFixed(1);
    if (data.gz !== undefined) elements.gyroZ.textContent = parseFloat(data.gz).toFixed(1);
}

// Update flex sensor display
function updateFlexSensor(name, value) {
    if (value === undefined) return;
    
    const val = parseFloat(value);
    const percent = Math.max(0, Math.min(100, val * 100));
    
    elements[`${name}Bar`].style.width = `${percent}%`;
    elements[`${name}Val`].textContent = (val * 100).toFixed(0) + '%';
}

// Update confidence display
function updateConfidence(confidence) {
    elements.confidenceFill.style.width = `${confidence}%`;
    elements.confidenceValue.textContent = `${confidence.toFixed(0)}%`;
}

// Add gesture to history
function addToHistory(gesture) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString();
    
    historyItems.unshift({ gesture, time: timeStr });
    
    if (historyItems.length > MAX_HISTORY) {
        historyItems.pop();
    }
    
    renderHistory();
}

// Render history list
function renderHistory() {
    if (historyItems.length === 0) {
        elements.historyList.innerHTML = '<div class="history-empty">Waiting for gestures...</div>';
        return;
    }
    
    elements.historyList.innerHTML = historyItems.map(item => `
        <div class="history-item">
            <span class="history-gesture">${item.gesture}</span>
            <span class="history-time">${item.time}</span>
        </div>
    `).join('');
}

// Calculate prediction rate
function updatePredRate() {
    const now = Date.now();
    const elapsed = (now - lastPredTime) / 1000;
    
    if (elapsed > 0) {
        const rate = predCount / elapsed;
        elements.predRate.textContent = rate.toFixed(1);
    }
    
    predCount = 0;
    lastPredTime = now;
}

// Fallback: Poll REST endpoint if WebSocket fails
let pollInterval = null;
function startPolling() {
    if (pollInterval) return;
    
    pollInterval = setInterval(async () => {
        try {
            const response = await fetch('/data');
            if (response.ok) {
                const data = await response.json();
                handleData(data);
            }
        } catch (e) {
            console.error('Polling error:', e);
        }
    }, 100); // 10 Hz polling
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

// Mode management
let currentMode = 'gesture'; // 'gesture' or 'sentence'
let sentencePollInterval = null;

// Switch to gesture mode
function switchToGestureMode() {
    if (currentMode === 'gesture') return;
    
    currentMode = 'gesture';
    console.log('Switched to gesture mode');
    
    // Stop sentence polling if active
    if (sentencePollInterval) {
        clearInterval(sentencePollInterval);
        sentencePollInterval = null;
    }
    
    // Update button states
    document.getElementById('gestureBtn').classList.add('active');
    document.getElementById('sentenceBtn').classList.remove('active');
}

// Switch to sentence mode
async function switchToSentenceMode() {
    if (currentMode === 'sentence') return;
    
    currentMode = 'sentence';
    console.log('Switched to sentence mode');
    
    // Update button states
    document.getElementById('gestureBtn').classList.remove('active');
    document.getElementById('sentenceBtn').classList.add('active');
    
    // Start continuous sentence predictions (trigger every 4.5 seconds)
    // Recording takes 4 seconds + 0.5 second gap = 4.5 seconds total
    triggerSentencePrediction(); // Immediate first trigger
    sentencePollInterval = setInterval(triggerSentencePrediction, 4500);
}

// Trigger a single sentence prediction
async function triggerSentencePrediction() {
    try {
        const response = await fetch('/api/sentence', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        if (!response.ok) {
            console.error(`HTTP ${response.status}: Failed to trigger sentence prediction`);
        }
    } catch (e) {
        console.error('Error triggering sentence:', e);
    }
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
    console.log('EchoSign Web UI initialized');
    
    // Get confidence bar element (parent of confidenceFill)
    elements.confidenceBar = elements.confidenceFill ? elements.confidenceFill.parentElement : null;
    
    // Initialize 3D hand model
    init3DHandModel();
    
    // Add mode button handlers
    const gestureBtn = document.getElementById('gestureBtn');
    const sentenceBtn = document.getElementById('sentenceBtn');
    const voiceToggle = document.getElementById('voiceToggle');
    
    if (gestureBtn) {
        gestureBtn.addEventListener('click', switchToGestureMode);
    }
    
    if (sentenceBtn) {
        sentenceBtn.addEventListener('click', switchToSentenceMode);
    }
    if (voiceToggle) {
        voiceToggle.addEventListener('change', (e) => {
            voiceEnabled = !!e.target.checked;
            if (voiceEnabled) {
                // Warm up voices list
                const _ = window.speechSynthesis.getVoices();
            } else {
                window.speechSynthesis.cancel();
            }
        });
    }
    
    // Start in gesture mode (default)
    currentMode = 'gesture';
    
    // Try WebSocket first
    connectWebSocket();
    
    // Start prediction rate calculation
    predRateInterval = setInterval(updatePredRate, 1000);
    
    // Fallback to polling after 5 seconds if no WebSocket
    setTimeout(() => {
        if (ws && ws.readyState !== WebSocket.OPEN) {
            console.log('WebSocket not connected, falling back to polling');
            startPolling();
        }
    }, 5000);
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (ws) ws.close();
    stopPolling();
    if (predRateInterval) clearInterval(predRateInterval);
});