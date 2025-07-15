'use client'
import { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

const classLabels = [
  {
    name: 'Dark',
    description: 'Bold, smoky flavor with low acidity and visible oils on the beans. Often has chocolate or caramel notes.'
  },
  {
    name: 'Green',
    description: 'Unroasted beans with grassy, herbal flavors. Higher acidity and caffeine content before roasting.'
  },
  {
    name: 'Light',
    description: 'Light brown color, no oil on beans. Bright acidity with floral/fruity notes and toasted grain flavors.'
  },
  {
    name: 'Medium',
    description: 'Balanced flavor with medium acidity and body. Shows caramel sweetness with nutty/chocolate undertones.'
  }
];

export default function Home() {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [prediction, setPrediction] = useState<{ name: string; description: string } | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsLoading(true);
        const loadedModel = await tf.loadLayersModel('/model/model.json');
        setModel(loadedModel);
      } catch (error) {
        console.error('Error loading model:', error);
        setError('Failed to load the model. Please try again later.');
      } finally {
        setIsLoading(false);
      }
    };
    loadModel();

    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      setError(null);
      setPrediction(null);
      setCapturedImage(null);
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setCameraActive(true);
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      setError("Could not access camera. Please check permissions and try again.");
    }
  };

  const captureImage = () => {
    if (videoRef.current && canvasRef.current) {
      const context = canvasRef.current.getContext('2d');
      if (context) {
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
        context.drawImage(
          videoRef.current, 
          0, 
          0, 
          canvasRef.current.width, 
          canvasRef.current.height
        );
        
        const imageDataURL = canvasRef.current.toDataURL('image/jpeg');
        setCapturedImage(imageDataURL);
        setCameraActive(false);
        
        // Stop camera stream after capture
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
          streamRef.current = null;
        }
      }
    }
  };

  const handlePredict = async () => {
    if (!model || !capturedImage) return;

    setIsLoading(true);
    const img = new Image();
    img.src = capturedImage;
    
    img.onload = async () => {
      try {
        const tensor = tf.tidy(() => {
          return tf.browser
            .fromPixels(img)
            .resizeNearestNeighbor([256, 256])
            .toFloat()
            .div(255.0)
            .expandDims();
        });

        const output = model.predict(tensor) as tf.Tensor;
        const data = await output.data();
        tensor.dispose();
        output.dispose();

        const predictedIndex = Array.from(data).indexOf(Math.max(...Array.from(data)));
        setPrediction(classLabels[predictedIndex]);
      } catch (error) {
        console.error('Prediction error:', error);
        setError('Error processing image. Please try again.');
      } finally {
        setIsLoading(false);
      }
    };
    
    img.onerror = () => {
      setError('Failed to load image. Please try again.');
      setIsLoading(false);
    };
  };

  const resetCamera = () => {
    setCapturedImage(null);
    setPrediction(null);
    setError(null);
    startCamera();
  };

  return (
    <div className="min-h-screen bg-amber-50 flex flex-col items-center p-6">
      <h1 className="text-3xl md:text-4xl font-bold text-amber-900 mb-8 mt-4 text-center">
        Coffee Roast Predictor ☕
      </h1>

      {/* Camera Preview */}
      {cameraActive && (
        <div className="w-full max-w-lg mb-6 relative">
          <video 
            ref={videoRef} 
            autoPlay 
            playsInline
            muted
            className="w-full h-auto rounded-xl shadow-lg border-4 border-amber-700"
          />
          <button
            onClick={captureImage}
            className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-red-600 hover:bg-red-700 text-white rounded-full p-4 shadow-lg"
          >
            <div className="w-16 h-16 rounded-full border-4 border-white flex items-center justify-center">
              <span className="sr-only">Capture</span>
            </div>
          </button>
        </div>
      )}

      {/* Captured Image Preview */}
      {capturedImage && (
        <div className="w-full max-w-lg mb-6 flex flex-col items-center">
          <img 
            src={capturedImage} 
            alt="Captured coffee beans"
            className="w-full h-auto rounded-xl shadow-lg border-4 border-amber-700"
          />
          <div className="flex space-x-4 mt-4">
            <button
              onClick={resetCamera}
              className="bg-amber-700 hover:bg-amber-800 text-white px-6 py-2 rounded-lg font-medium shadow transition"
            >
              Retake
            </button>
            <button
              onClick={handlePredict}
              disabled={isLoading}
              className={`${
                isLoading 
                  ? 'bg-gray-500 cursor-not-allowed' 
                  : 'bg-green-700 hover:bg-green-800'
              } text-white px-6 py-2 rounded-lg font-medium shadow transition`}
            >
              {isLoading ? 'Analyzing...' : 'Predict Roast'}
            </button>
          </div>
        </div>
      )}

      {/* Camera Activation */}
      {!cameraActive && !capturedImage && (
        <div className="w-full max-w-lg mb-8 flex flex-col items-center">
          <div className="bg-gray-200 border-2 border-amber-300 border-dashed rounded-xl w-full h-64 flex items-center justify-center mb-6">
            <div className="text-center p-4">
              <svg className="w-12 h-12 mx-auto text-amber-700 mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"></path>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"></path>
              </svg>
              <p className="text-amber-800 font-medium">Camera preview will appear here</p>
              <p className="text-amber-600 text-sm mt-2">Point at coffee beans and capture an image</p>
            </div>
          </div>
          <button
            onClick={startCamera}
            disabled={isLoading}
            className={`${
              isLoading 
                ? 'bg-gray-500 cursor-not-allowed' 
                : 'bg-amber-700 hover:bg-amber-800'
            } text-white px-8 py-3 rounded-lg text-lg font-semibold shadow-lg transition w-full max-w-xs`}
          >
            {isLoading ? 'Loading Model...' : 'Start Camera'}
          </button>
        </div>
      )}

      {/* Prediction Result */}
      {prediction && (
        <div className="w-full max-w-lg bg-amber-100 rounded-xl p-6 shadow-lg border border-amber-300 mt-4">
          <h2 className="text-2xl font-bold text-amber-900 mb-4">
            Roast Level: <span className="text-amber-700">{prediction.name}</span>
          </h2>
          <div className="bg-white p-4 rounded-lg shadow-inner">
            <h3 className="font-semibold text-amber-800 mb-2">Flavor Characteristics:</h3>
            <p className="text-amber-900">{prediction.description}</p>
          </div>
        </div>
      )}

      {/* Hidden canvas for image capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Loading Indicator */}
      {isLoading && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-8 flex flex-col items-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-amber-700 mb-4"></div>
            <p className="text-lg font-medium text-gray-700">Analyzing coffee beans...</p>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="w-full max-w-lg mt-4 p-3 bg-red-100 text-red-700 rounded-lg text-center">
          {error}
        </div>
      )}

      <footer className="mt-8 text-center text-amber-700 text-sm">
        <p>Point your camera at coffee beans to analyze their roast level</p>
        <p className="mt-2">Make sure beans are well-lit and fill most of the frame</p>
      </footer>
    </div>
  );
}