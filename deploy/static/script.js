// static/script.js (Versi Baru)

const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');

const leftEyeStatusElem = document.getElementById('left-eye-status');
const rightEyeStatusElem = document.getElementById('right-eye-status');
const yawnStatusElem = document.getElementById('yawn-status');
const pitchStatusElem = document.getElementById('pitch-status');
const yawStatusElem = document.getElementById('yaw-status');
const rollStatusElem = document.getElementById('roll-status');
const alertBox = document.getElementById('alert-box');

const EYES_CLOSED_FRAMES_THRESHOLD = 15;
const HEAD_DOWN_FRAMES_THRESHOLD = 20;
const PITCH_THRESHOLD_DEG = -20;
const MAR_THRESHOLD = 0.5;
const YAWN_CONSECUTIVE_FRAMES = 5;

let eyesClosedCounter = 0;
let headDownCounter = 0;
let yawnCounter = 0;

async function startDetection() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 }
            }
        });
        video.srcObject = stream;
        video.addEventListener('loadeddata', () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            setInterval(detectFrame, 150);
        });
    } catch (err) {
        console.error("Error accessing webcam: ", err);
        alert("Could not access the webcam. Please ensure it's enabled and permission is granted.");
    }
}

async function detectFrame() {
    if (video.paused || video.ended) {
        return;
    }

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    const imageData = tempCanvas.toDataURL('image/jpeg', 0.8); 

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData }),
        });
        const data = await response.json();
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (data.face_found) {
            drawResults(data);
            updateStatus(data);
            checkDrowsiness(data);
        } else {
            resetCounters();
            updateStatus({}); 
        }
    } catch (error) {
        console.error('Error during detection:', error);
    }
}


function drawResults(data) {
    // Draw face box
    ctx.strokeStyle = 'lime';
    ctx.lineWidth = 2;
    const [fx, fy, fmaxx, fmaxy] = data.face_box;
    ctx.strokeRect(fx, fy, fmaxx - fx, fmaxy - fy);
    
    // Draw eye boxes
    data.eye_boxes.forEach((box, i) => {
        const status = data.eye_statuses[i];
        ctx.strokeStyle = (status === 'Terbuka') ? 'cyan' : 'red';
        const [ex, ey, emaxx, emaxy] = box;
        ctx.strokeRect(ex, ey, emaxx - ex, emaxy - ex);
    });

    // Draw head pose axes
    if (data.head_pose) {
        drawAxes(data.head_pose, data.face_box);
    }
}

function drawAxes(pose, faceBox) {
    const { pitch, yaw, roll } = pose;
    const [fx, fy, fmaxx, fmaxy] = faceBox;
    const nose_2d = { x: (fx + fmaxx) / 2, y: (fy + fmaxy) / 2 };
    
    const pitch_rad = pitch * Math.PI / 180;
    const yaw_rad = -(yaw * Math.PI / 180);
    const roll_rad = roll * Math.PI / 180;
    const size = 50;

    // Simplified 2D projection of 3D axes
    const p1 = { x: nose_2d.x, y: nose_2d.y };
    const axisX = { x: p1.x + size * (Math.cos(yaw_rad) * Math.cos(roll_rad)), y: p1.y + size * (Math.cos(pitch_rad) * Math.sin(roll_rad) + Math.cos(roll_rad) * Math.sin(pitch_rad) * Math.sin(yaw_rad)) };
    const axisY = { x: p1.x + size * (-Math.cos(yaw_rad) * Math.sin(roll_rad)), y: p1.y + size * (Math.cos(pitch_rad) * Math.cos(roll_rad) - Math.sin(pitch_rad) * Math.sin(yaw_rad) * Math.sin(roll_rad)) };
    const axisZ = { x: p1.x + size * (Math.sin(yaw_rad)), y: p1.y - size * (Math.cos(yaw_rad) * Math.sin(pitch_rad)) };

    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(axisX.x, axisX.y);
    ctx.strokeStyle = 'red'; // X-axis
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(axisY.x, axisY.y);
    ctx.strokeStyle = 'green'; // Y-axis
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(axisZ.x, axisZ.y);
    ctx.strokeStyle = 'blue'; // Z-axis
    ctx.stroke();
}


function updateStatus(data) {
    leftEyeStatusElem.textContent = data.eye_statuses?.[0] || 'N/A';
    rightEyeStatusElem.textContent = data.eye_statuses?.[1] || 'N/A';
    pitchStatusElem.textContent = data.head_pose?.pitch.toFixed(2) || '0.0';
    yawStatusElem.textContent = data.head_pose?.yaw.toFixed(2) || '0.0';
    rollStatusElem.textContent = data.head_pose?.roll.toFixed(2) || '0.0';
}

function resetCounters() {
    eyesClosedCounter = 0;
    headDownCounter = 0;
    yawnCounter = 0;
}


function checkDrowsiness(data) {
    let isDrowsy = false;
    let reason = "";

    // Check for closed eyes
    if (data.eye_statuses && data.eye_statuses.every(s => s === 'Tertutup')) {
        eyesClosedCounter++;
    } else {
        eyesClosedCounter = 0;
    }

    if (eyesClosedCounter >= EYES_CLOSED_FRAMES_THRESHOLD) {
        isDrowsy = true;
        reason = "Mata Tertutup";
    }

    // Check for head down
    if (data.head_pose && data.head_pose.pitch < PITCH_THRESHOLD_DEG) {
        headDownCounter++;
    } else {
        headDownCounter = 0;
    }

    if (headDownCounter >= HEAD_DOWN_FRAMES_THRESHOLD) {
        isDrowsy = true;
        reason = "Kepala Terkulai";
    }
    
    // Check for yawning
    if (data.mar > MAR_THRESHOLD) {
        yawnCounter++;
    } else {
        yawnCounter = 0;
    }
    
    if (yawnCounter >= YAWN_CONSECUTIVE_FRAMES) {
        yawnStatusElem.textContent = "MENGUAP";
        isDrowsy = true;
        reason = "Menguap";
    } else {
        yawnStatusElem.textContent = "NORMAL";
    }

    // Update alert box
    if (isDrowsy) {
        alertBox.textContent = `!!! PERINGATAN KANTUK (${reason}) !!!`;
        alertBox.className = 'alert-on';
    } else {
        alertBox.textContent = 'NO ALERT';
        alertBox.className = 'alert-off';
    }
}


startDetection();