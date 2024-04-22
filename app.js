const express = require("express");
const faceapi = require("./js/face-api");
const canvas = require("canvas");
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });
const fetch = require("node-fetch");
const app = express();
app.use(express.static("public"));
PORT = 4000;

async function preloadModels() {
  await faceapi.nets.tinyFaceDetector.loadFromDisk("./models");
  await faceapi.nets.faceLandmark68Net.loadFromDisk("./models");
  await faceapi.nets.faceRecognitionNet.loadFromDisk("./models");
}

async function getDescriptors(imageFile) {
  const faces = await faceapi
    .detectAllFaces(imageFile, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks()
    .withFaceDescriptors();

  faceapi.tf.dispose(imageFile);
  return faces.map((face) => face.descriptor);
}

async function loadAndProcessImage(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch image (HTTP status ${response.status})`);
  }
  const buffer = await response.buffer();
  const img = new Image();
  img.src = `data:image/jpeg;base64,${buffer.toString("base64")}`;
  return img;
}

async function main(faceUrl1, faceUrl2) {
  const [img1, img2] = await Promise.all([
    loadAndProcessImage(faceUrl1),
    loadAndProcessImage(faceUrl2),
  ]);
  const [desc1, desc2] = await Promise.all([
    getDescriptors(img1),
    getDescriptors(img2),
  ]);
  const distance = faceapi.euclideanDistance(desc1[0], desc2[0]);
  return (1 - distance) * 100;
}

app.get("/", (req, res) => {
  res.send("Welcome to Face Recognition");
});

app.get("/face", async (req, res) => {
  const { face1, face2 } = req.query;
  console.log("Face 1:", face1);
  console.log("Face 2:", face2);
  console.time("Processing Time");
  const distance = await main(face1, face2);
  console.timeEnd("Processing Time");
  res.send({ faceMatch: distance });
  console.log("Face Match Percentage:", distance);
});

(async () => {
  await preloadModels();
  app.listen(PORT, () => {
    console.log(`Server is running on port: http://localhost:${PORT}`);
  });
})();

// app.listen(PORT, (error) => {
//   if (!error)
//     console.log(
//       "Server is Successfully Running,and App is listening on port " + PORT
//     );
//   else console.log("Error occurred, server can't start", error);
// });
