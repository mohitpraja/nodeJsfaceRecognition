const express = require("express");
const faceapi = require("./js/face-api");
const canvas = require("canvas");
const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });
const fetch = require("node-fetch");
const app = express();
app.use(express.static("public"));
PORT = 4000;

const getDescriptors = async (imageFile) => {
  // const buffer = fs.readFileSync(imageFile);
  // const tensor = faceapi.tf.node.decodeImage(buffer, 3);
  const faces = await faceapi
    .detectAllFaces(imageFile, new faceapi.TinyFaceDetectorOptions())
    .withFaceLandmarks()
    .withFaceDescriptors();

  faceapi.tf.dispose(imageFile);
  return faces.map((face) => face.descriptor);
};

const main = async (file1, file2) => {
  console.log("input images:", file1, file2); // eslint-disable-line no-console
  ////////////////////////////////////////////////////////////////////////
  const response = await fetch(file1);
  const response2 = await fetch(file2);

  // Ensure response is OK (status code 200)
  if (!response.ok) {
    throw new Error(`Failed to fetch image (HTTP status ${response.status})`);
  }

  // Read image data as buffer
  const buffer = await response.buffer();
  const buffer2 = await response2.buffer();

  // Create an HTMLImageElement
  const img = new Image();
  const img2 = new Image();

  // Set the image source to the buffer (Base64 data URL)
  img.src = `data:image/jpeg;base64,${buffer.toString("base64")}`;
  img2.src = `data:image/jpeg;base64,${buffer2.toString("base64")}`;
  ////////////////////////////////////////////////////////////////////////
  await faceapi.tf.ready();
  const [desc1, desc2] = await Promise.all([
    getDescriptors(img),
    getDescriptors(img2),
  ]);
  const distance = faceapi.euclideanDistance(desc1[0], desc2[0]); // only compare first found face in each image
  return (1 - distance) * 100;
};
app.get("/", (req, res) => {
  res.send("Welcome to Face Recognition");
});
app.get("/face", async (req, res) => {
  const { face1, face2 } = req.query; // Access query parameters using req.query
  console.log("Face 1:", face1); // Log the value of 'face1'
  console.log("Face 2:", face2); // Log the value of 'face2'
  console.time("add");
  var dist = await main(face1, face2);
  console.timeEnd("add");
  res.send({ faceMatch: dist });
  console.log(dist);
});
async function loadModels() {
  
    await faceapi.nets.tinyFaceDetector.loadFromDisk("./models"),
    await faceapi.nets.faceExpressionNet.loadFromDisk("./models"),
    await faceapi.nets.ssdMobilenetv1.loadFromDisk("./models");
    await faceapi.nets.faceLandmark68Net.loadFromDisk("./models");
    await faceapi.nets.faceRecognitionNet.loadFromDisk("./models");
}

// Call loadModels when your server starts
loadModels()
  .then(() => {
    console.log("Face models loaded successfully.");
  })
  .catch((error) => {
    console.error("Failed to load face models:", error);
  });

async function loadImage(url) {
  const image = await canvas.loadImage(url);
  return image;
}

async function recognizeFaces(img, img2) {
  const image1 = await loadImage(img);
  const image2 = await loadImage(img2);
  const detectionOptions = new faceapi.SsdMobilenetv1Options({
    minConfidence: 0.5,
    maxResults: 1,
  });

  const [detections1, detections2] = await Promise.all([
    faceapi
      .detectAllFaces(image1, detectionOptions)
      .withFaceLandmarks()
      .withFaceDescriptors(),
    faceapi
      .detectAllFaces(image2, detectionOptions)
      .withFaceLandmarks()
      .withFaceDescriptors(),
  ]);

  if (detections1.length > 0 && detections2.length > 0) {
    const distance = faceapi.euclideanDistance(
      detections1[0].descriptor,
      detections2[0].descriptor
    );
    const similarity = 1 - distance;
    console.log("Similarity between faces:", similarity);
  } else {
    console.log("No faces detected in one or both images.");
  }
}

app.get("/face1", async (res, req) => {
  recognizeFaces(
    "https://shakir.ubihrm.com/public/attendance_images/10/358044/2024-04-08/40003_20240408_0957_38.jpg",
    "https://ubiattendanceimages.s3.ap-south-1.amazonaws.com/public/attendance_images/10/358044/2024-04-16/40003_20240416_1026_1.jpg"
  );
});

app.listen(PORT, (error) => {
  if (!error)
    console.log(
      "Server is Successfully Running,and App is listening on port " + PORT
    );
  else console.log("Error occurred, server can't start", error);
});
