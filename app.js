const express = require('express')
const faceapi = require("./js/face-api");
const canvas = require('canvas');
const { Canvas, Image, ImageData } = canvas;
 faceapi.env.monkeyPatch({ Canvas, Image, ImageData });
const fetch = require('node-fetch');
const app = express()
PORT= 4000;

const getDescriptors = async (imageFile) => {
  // const buffer = fs.readFileSync(imageFile);
  // const tensor = faceapi.tf.node.decodeImage(buffer, 3);
  const faces = await faceapi
        .detectAllFaces(imageFile, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks()
        .withFaceDescriptors();

        
    faceapi.tf.dispose(imageFile);
  return faces.map((face) => face.descriptor);
};

const main = async (file1, file2) => {
  console.log('input images:', file1, file2); // eslint-disable-line no-console
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
  img.src = `data:image/jpeg;base64,${buffer.toString('base64')}`;
  img2.src = `data:image/jpeg;base64,${buffer2.toString('base64')}`;
  ////////////////////////////////////////////////////////////////////////
  await faceapi.tf.ready();
  const [desc1, desc2] = await Promise.all([
    getDescriptors(img),
    getDescriptors(img2)
  ]);
  const distance = faceapi.euclideanDistance(desc1[0], desc2[0]); // only compare first found face in each image
  return (1 - distance)*100;
};
app.get('/',(req,res)=>{
  res.send("Welcome to Face Recognition")
})
app.get("/face", async(req, res) =>{ 
  const { face1, face2 } = req.query; // Access query parameters using req.query
  console.log("Face 1:", face1); // Log the value of 'face1'
  console.log("Face 2:", face2); // Log the value of 'face2'
    console.time("add");
    var dist = await main(face1, face2);
    console.timeEnd("add");
    res.send({faceMatch:dist});
    console.log(dist);
  });



app.get('/face_similarity', async (req, res)=>{
    res.status(200); 
    res.send("Welcome to root URL of Server"); 

    await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromDisk("models"),
        faceapi.nets.tinyYolov2.loadFromDisk("models"),
        faceapi.nets.ssdMobilenetv1.loadFromDisk("models"),
        faceapi.nets.faceLandmark68Net.loadFromDisk("models"),
        faceapi.nets.faceRecognitionNet.loadFromDisk("models"),
        // faceapi.nets.ageGenderNet.loadFromDisk("./models"),
      ]);

      console.log(faceapi.nets)

    const url = 'https://shakir.ubihrm.com/public/attendance_images/10/358044/2024-04-08/40003_20240408_0957_38.jpg';
    
    const url2 = 'https://shakir.ubihrm.com/public/attendance_images/85218/2275650/2024-04-10/40003_20240410_0911_25.jpg';
    // const url2 = 'https://shakir.ubihrm.com/public/attendance_images/90303/342265/2024-04-08/40003_20240408_0033_6.jpg';


    urlToImageElement(url,url2)
  .then((imageElement) => {
    if (imageElement) {
       console.log('Image loaded successfully:');
      // Now you can use the imageElement for rendering or further processing
    } else {
      console.log('Failed to load image.');
    }
  })
  .catch((error) => {
    console.error('Error loading image:', error);
  });

    
  return


    // var descriptors = await faceapi.computeFaceDescriptor(img) 
    console.log("check ");
    // console.log(descriptors);




// console.log("returnedB64"); 
// console.log(typeof returnedB64);

//     request.get('https://shakir.ubihrm.com/public/attendance_images/10/358044/2024-04-08/40003_20240408_0957_38.jpg', function (error, response, body) {
//     if (!error && response.statusCode == 200) {
//         data = "data:" + response.headers["content-type"] + ";base64," + Buffer.from(body).toString('base64');
//         console.log("data-4643");
//         console.log(typeof data);
//     }
// });

return

request.get({url : url, encoding: null}, async (err, res, body) => {
    if (!err) {
        const type = res.headers["content-type"];
        const prefix = "data:" + type + ";base64,";
        const base64 = body.toString('base64');
        const dataUri = prefix + base64;
        console.log("dataUri");

        const buffer = Buffer.from(url, 'base64');
        console.log(buffer);
// Step 2: Convert the buffer to a JSON object
const jsonObject = JSON.parse(buffer.toString('utf-8'));

console.log(jsonObject);
        
return
        // console.log(dataUri);
        // console.log(typeof JSON.parse(dataUri));
        // let objJsonStr = JSON.stringify(dataUri);
        // encode a string
        // const decodedData = atob(dataUri); // decode the string
        // console.log(encodedData);
        // console.log(decodedData);

        var descriptors = await faceapi.computeFaceDescriptor(dataUri) 
        console.log("check ");
        console.log(descriptors);

    }
});

// imageToBase64("https://shakir.ubihrm.com/public/attendance_images/10/358044/2024-04-08/40003_20240408_0957_38.jpg") // insert image url here. 
//     .then( (response) => {
//           console.log(response);  // the response will be the string base64.
//       }
//     )
//     .catch(
//         (error) => {
//             console.log(error);
//         }
//     )

    // const threshold = 0.6
    // let descriptors = { desc1: null, desc2: null }
    // const input = await faceapi.fetchImage("https://shakir.ubihrm.com/public/attendance_images/10/358044/2024-04-08/40003_20240408_0957_38.jpg")
    // console.log("input");
    // console.log(input);
    //   const imgEl = $(`#face${which}`).get(0)
    //   imgEl.src = input.src
    //   console.log("which-60");
    //   console.log(which);
    //   console.log(uri);
    //   descriptors[`desc${which}`] = await faceapi.computeFaceDescriptor(input)
    // res.sendFile(path.join(viewsDir, 'bbtFaceSimilarity.html'))
});


async function urlToImageElement(imageUrl,Imag2) {
    try {
      // Fetch the image from the URL
      console.log("function to call this");
      const response = await fetch(imageUrl);
      const response2 = await fetch(Imag2);
  
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
      img.src = `data:image/jpeg;base64,${buffer.toString('base64')}`;
      img2.src = `data:image/jpeg;base64,${buffer2.toString('base64')}`;
      console.log("img-131313");
    //   console.log(img);
    const useTinyModel = true
    // const detectionsWithLandmarks = await faceapi.detectAllFaces(input).withFaceLandmarks(useTinyModel)

    // let refFaceAiData = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors()
    // let faceMatcher = new faceapi.FaceMatcher(refFaceAiData)

    // Minimum face size for detection



    const refFaceAiData = await faceapi.tf.tidy(() => {
      // const resizedImg = faceapi.resizeResults(img, { width: 640 });
      let minSize = 200; 
      return faceapi.detectAllFaces(img, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5, minFaceSize: minSize })).withFaceLandmarks().withFaceDescriptors()
  });

  let faceMatcher = new faceapi.FaceMatcher(refFaceAiData)
    console.log("faceMatcher");
    console.log(refFaceAiData);

    const results = await faceapi.tf.tidy(() => {
      let minSize = 200; 
      // const resizedImg = faceapi.resizeResults(img, { width: 640 });
      return faceapi.detectAllFaces(img2,new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5, minFaceSize: minSize })).withFaceLandmarks().withFaceDescriptors();
  });

  //   const results = await faceapi
  //   .detectAllFaces(img2)
  //   .withFaceLandmarks()
  //   .withFaceDescriptors()
  
  results.forEach(fd => {
    const bestMatch = faceMatcher.findBestMatch(fd.descriptor)
    console.log("bestMatch.toString()")
    console.log(bestMatch.toString())
  })

return
    // let facesToCheckAiData = await faceapi.detectAllFaces(img2).withFaceLandmarks(useTinyModel).withFaceDescriptors()
    // console.log(refFaceAiData);
    // console.log("shakir1");
    // console.log(facesToCheckAiData);
    //   var descriptors1 = await faceapi.computeFaceDescriptor(img) 
    //   var descriptors2 = await faceapi.computeFaceDescriptor(img2) 
    //     console.log(descriptors1);
    //     console.log(descriptors2);
    //     console.log("descriptors2");
    //     const threshold = 0.2
    //     const distance = faceapi.utils.round(
    //         faceapi.euclideanDistance(descriptors1, descriptors2)
    //       )
    //       let text = distance
         
    //       if (distance > threshold) {
    //     console.log(text +  ' (no match)')
    //        // bgColor = '#ce7575'
    //       }else{
    //         console.log(text +  ' (match)')
    //       }

       

        facesToCheckAiData = faceapi.resizeResults(facesToCheckAiData,img2)
        console.log("shakir2");
        //loop through all of hte faces in our imageToCheck and compare to our reference datta
        facesToCheckAiData.forEach(face=>{
            const { detection, descriptor } = face
            //make a label, using the default
            console.log("shakir");
            let label = faceMatcher.findBestMatch(descriptor).toString()
            console.log("label-64641")
            console.log(label)
            if(label.includes("unknown")){
                return
               
            }
            let options = { label: "Jordan"}
            // const drawBox = new faceapi.draw.DrawBox(detection.box,options)
            // drawBox.draw(canvas)
        })


      // Wait for the image to load by wrapping it in a Promise
    //   await new Promise((resolve, reject) => {
    //     img.onload = () => resolve();
    //     img.onerror = (error) => reject(error);
    //   });
  
      return img; // Return the loaded HTMLImageElement
    } catch (error) {
      console.error('Error:', error);
      return null; // Return null if an error occurs
    }
  }

// res.sendFile(path.join(viewsDir, 'bbtFaceSimilarity.html')))
// app.get('/face_similarity', (req, res)=>res.sendFile(path.join(viewsDir, 'bbtFaceSimilarity.html')))

async function loadModels() {
  await faceapi.nets.tinyFaceDetector.loadFromDisk("./models"),
  await faceapi.nets.faceExpressionNet.loadFromDisk("./models"),
  //faceapi.nets.tinyYolov2.loadFromDisk("./models"),
  await faceapi.nets.ssdMobilenetv1.loadFromDisk('./models');
  await faceapi.nets.faceLandmark68Net.loadFromDisk('./models');
  await faceapi.nets.faceRecognitionNet.loadFromDisk('./models');
}

// Call loadModels when your server starts
loadModels().then(() => {
  console.log('Face models loaded successfully.');
}).catch((error) => {
  console.error('Failed to load face models:', error);
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
    maxResults: 1
});

  const [detections1, detections2] = await Promise.all([
      faceapi.detectAllFaces(image1, detectionOptions).withFaceLandmarks().withFaceDescriptors(),
      faceapi.detectAllFaces(image2, detectionOptions).withFaceLandmarks().withFaceDescriptors()
  ]);

  if (detections1.length > 0 && detections2.length > 0) {
      const distance = faceapi.euclideanDistance(detections1[0].descriptor, detections2[0].descriptor);
      const similarity = 1 - distance;
      console.log('Similarity between faces:', similarity);
  } else {
      console.log('No faces detected in one or both images.');
  }
}

app.get('/face1',async (res,req)=>{
  recognizeFaces("https://shakir.ubihrm.com/public/attendance_images/10/358044/2024-04-08/40003_20240408_0957_38.jpg", "https://ubiattendanceimages.s3.ap-south-1.amazonaws.com/public/attendance_images/10/358044/2024-04-16/40003_20240416_1026_1.jpg");
});


app.listen(PORT, (error) =>{ 
    if(!error) 
        console.log("Server is Successfully Running,and App is listening on port "+ PORT) 
    else 
        console.log("Error occurred, server can't start", error); 
    });