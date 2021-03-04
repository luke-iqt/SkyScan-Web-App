import React, { useState, useRef, useEffect, useReducer } from "react";
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import "./App.css";

const machine = {
  initial: "initial",
  states: {
    initial: { on: { next: "startWebcam" } },
    startWebcam: {on: { next: "loadModel"}},
    loadModel: { on: { next: "identify" } },
    identify: { on: { next: "complete" }},
    complete: { on: { next: "identify" }, showImage: true, showResults: true }
  }
};


let classList = {
  1: {
      name: 'Plane',
      id: 1,
  }
}
const threshold = 0.60;


function App() {
  const [results, setResults] = useState([]);
  const [model, setModel] = useState(null);
  const [modelURL, setModelURL] = useState('web_mobilenet_balanced_model/model.json');
  const [webcam, setWebcam] = useState(null);
  const [showWebcam, setShowWebcam] = useState(false);
  const [objPred, setObjPred] = useState(null);
  const [modelReady, setModelReady] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const videoRef = useRef();
  const canvasRef = useRef();

 
  const reducer = (state, event) => 
    machine.states[state].on[event] || machine.initial;
  const [appState, dispatch] = useReducer(reducer, machine.initial);
  const next = () => dispatch("next");

  const loadModel = async () => {
    console.log('Loading ' + modelURL + '...');
  
    const model = await loadGraphModel(modelURL);
    
    console.log('Successfully loaded model');
    setModel(model);
    setModelReady(true);
  };

  
  const startWebcam = async () => {
    setShowWebcam(true);
    if ('mediaDevices' in navigator && 'getUserMedia' in navigator.mediaDevices) {
      console.log("Let's get this party started")
    }
    navigator.mediaDevices.getUserMedia({video: true})
  
    const devices = await navigator.mediaDevices.enumerateDevices();
    console.log(devices);
    const webcamConfig = { resizeWidth: 300, resizeHeight: 300, centerCrop: true, facingMode: 'environment'}
    const webcam = await tf.data.webcam(videoRef.current,webcamConfig);
    setWebcam(webcam);
    if (!modelReady) {
      loadModel();
    }
  };

  const buildDetectedObjects = (scores, threshold, boxes, classes, classesDir) => {
    const detectionObjects = []
    var video_frame = document.getElementById('webcam');

    scores[0].forEach((score, i) => {
      if (score > threshold) {
        const bbox = [];
        const minY = boxes[0][i][0] * 300;//video_frame.offsetHeight;
        const minX = boxes[0][i][1] * 300;//video_frame.offsetWidth;
        const maxY = boxes[0][i][2] * 300;//video_frame.offsetHeight;
        const maxX = boxes[0][i][3] * 300;//video_frame.offsetWidth;
        bbox[0] = minX;
        bbox[1] = minY;
        bbox[2] = maxX - minX;
        bbox[3] = maxY - minY;
        const classNum = classes[0][i] 
        var label = "Thing";
        if (classesDir[classNum]) {
          label = classesDir[classNum].name
        }
        detectionObjects.push({
          class: classNum,
          label: label,
          score: score.toFixed(4),
          bbox: bbox
        })
      }
    })
    return detectionObjects
  }
  const renderPredictions = predictions => {
    const ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    //Getting predictions

    /*
    // MobileNet FPN
    const  classes = predictions[2].arraySync() //7
    const scores = predictions[4].arraySync()  //4
    const boxes = predictions[3].arraySync()  //6
*/

    // MobileNet 
    const  classes = predictions[0].arraySync() //7
    const scores = predictions[1].arraySync()  //1 //4
    const boxes = predictions[4].arraySync()  //6
/*
    const prediction0 = predictions[0].arraySync()
    const prediction1 = predictions[1].arraySync()
    const prediction2 = predictions[2].arraySync()
    const prediction3 = predictions[3].arraySync()
    const prediction4 = predictions[4].arraySync()
    const prediction5 = predictions[5].arraySync()
    const prediction6 = predictions[6].arraySync()
    const prediction7 = predictions[7].arraySync()
    
    */
    const detections = buildDetectedObjects(scores, threshold,
      boxes, classes, classList);

    detections.forEach(item => {
      const x = item['bbox'][0];
      const y = item['bbox'][1];
      const width = item['bbox'][2];
      const height = item['bbox'][3];

      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x, y, width, height);

      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(item["label"] + " " + (100 * item["score"]).toFixed(2) + "%").width;
      const textHeight = parseInt(font, 10); // base 10
      ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
    });

    detections.forEach(item => {
      const x = item['bbox'][0];
      const y = item['bbox'][1];

      // Draw the text last to ensure it's on top.
      ctx.fillStyle = "#000000";
      ctx.fillText(item["label"] + " " + (100*item["score"]).toFixed(2) + "%", x, y);
    });
  };

  const detect = async () => {
    const img = await webcam.capture();
    let tensor = img.reshape([1,300, 300,3]).toInt(); // change the image size


    let offset = tf.scalar(127.5);
    var  new_frame = img.expandDims().reshape([1, 300, 300, 3]);    
    var test_frame = tf.expandDims(img.toInt()).reshape([-1, 300, 300, 3]);

    const predictions = await model.executeAsync(tensor); 
    renderPredictions(predictions)
  };

  const reset = async () => {
    setResults([]);
    next();
  };


  const actionButton = {
    initial: { action: startWebcam, text: "Start" },
    startWebcam: { text: "Starting Webcam..." },
    loadModel: { text: "Loading Model..." },
    identify: { text: "Identifying..." },
    complete: { action: reset, text: "Reset" }
  };

 
  useEffect(() => {
    const timeout = setInterval(() => {
      if (modelReady && webcam) {
        detect();
      }
    }, 1000);
    return () => {
      clearInterval(timeout);  // this guarantees to run right before the next effect
    }
  });

  useEffect(() => {
    loadModel();
  }, [modelURL]); // Only re-run the effect if count changes
  
  return (
    <div>
    <div id="main">
          <h1>Plane Spotter</h1>
          {!showWebcam && (
      <button onClick={startWebcam}>
        Start Webcam
      </button>
      )}
      {showWebcam && (
       <div id="video-box"  >  
      <video autoPlay playsInline muted id="webcam" width="300px" height="300px" style={{objectFit:"cover"}} ref={videoRef} />
      <canvas
          className="size"
          ref={canvasRef}
        />
      </div> 
      )}
   
       <div id="disclaimer"><p>This model was trained on airplanes flying by at around 30,000 feet. It works best at identifying small plane silhouettes.</p></div>
      </div>
      <div id="explainer">
      <div>
        <h2>What is this?</h2>
        <p>An easy way to try out a custom Object Detection model that was trained using data collected by the <a href="https://github.com/IQTLabs/SkyScan">SkyScan</a> system.</p>
       <h2>How did you build this?</h2>
        <p><a href="https://github.com/IQTLabs/SkyScan">SkyScan</a> takes photos of airplanes as they fly by. It determines a planes location based
        on the location signal they broadcast, using a standard called ADS-B. When a plane is nearby, it will point a camera at the airplane
         zoom in and take a photo. This gave us a lot of data that was easy to label, so we used it to train a model.</p>
       <p>We have created a series of <a href="https://github.com/IQTLabs/SkyScan/tree/main/ml-model">notebooks</a> that make it easy to train your own Object Detections model using the TensorFlow 2.0 <a href="https://github.com/tensorflow/models/tree/master/research/object_detection">Object Detection API</a>.</p>

        <h3>Browser based apps are a great way to find out how well your model works in the real world!</h3>
      </div>
      </div>
      </div>
    
  );
}

export default App;