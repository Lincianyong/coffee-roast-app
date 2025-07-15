'use client';
import { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

const classLabels = [
  {
    name: 'Dark',
    description: 'Rasa kuat dan berasap dengan keasaman rendah dan permukaan biji yang berminyak. Sering memiliki aroma cokelat atau karamel.',
  },
  {
    name: 'Green',
    description: 'Biji yang belum disangrai dengan rasa seperti rumput atau herbal. Memiliki tingkat keasaman dan kafein yang lebih tinggi sebelum disangrai.',
  },
  {
    name: 'Light',
    description: 'Berwarna cokelat terang, tanpa minyak di permukaan biji. Keasaman yang cerah dengan aroma bunga/buah dan rasa seperti biji-bijian panggang.',
  },
  {
    name: 'Medium',
    description: 'Rasa seimbang dengan keasaman dan body sedang. Memiliki rasa manis karamel dengan sentuhan kacang atau cokelat.',
  }
];

export default function Home() {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [prediction, setPrediction] = useState<{ name: string; description: string } | null>(null);
  const [imageURL, setImageURL] = useState<string | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        const loadedModel = await tf.loadLayersModel('/model/model.json');
        setModel(loadedModel);
      } catch (err) {
        console.error(err);
        setError('Failed to load model. Please refresh and try again.');
      } finally {
        setIsModelLoading(false);
      }
    };
    loadModel();

    // Cleanup camera on unmount
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      setPrediction(null);
      setError(null);
      setImageURL(null);
      
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { facingMode: 'environment' } 
      });
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setCameraActive(true);
      }
    } catch (err) {
      console.error('Camera error:', err);
      setError('Camera access failed. Please ensure you have granted camera permissions.');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    setCameraActive(false);
  };

  const captureImage = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw current video frame to canvas
    context?.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Get image URL from canvas
    const url = canvas.toDataURL('image/jpeg');
    setImageURL(url);
    stopCamera();
  };

  const handlePredict = async () => {
    if (!model || !imageURL) return;

    setIsPredicting(true);
    setError(null);

    const img = new Image();
    img.src = imageURL;
    img.crossOrigin = 'anonymous';

    img.onload = async () => {
      try {
        const tensor = tf.tidy(() => {
          return tf.browser
            .fromPixels(img)
            .resizeBilinear([256, 256])
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
      } catch (err) {
        console.error(err);
        setError('Prediction failed. Please try again.');
      } finally {
        setIsPredicting(false);
      }
    };

    img.onerror = () => {
      setError('Failed to load image.');
      setIsPredicting(false);
    };
  };

  return (
    <div className="min-h-screen bg-amber-50 flex flex-col items-center p-6">
      <h1 className="text-3xl md:text-4xl font-bold text-amber-900 mb-8 mt-4 text-center">
        Coffee Roast Predictor ☕
      </h1>

      <div className="w-full max-w-lg flex flex-col items-center">
        {!cameraActive ? (
          <button
            onClick={startCamera}
            className="bg-amber-700 hover:bg-amber-800 text-white font-medium px-6 py-3 rounded-lg shadow mb-6"
          >
            Open Camera
          </button>
        ) : (
          <button
            onClick={stopCamera}
            className="bg-red-600 hover:bg-red-700 text-white font-medium px-6 py-3 rounded-lg shadow mb-6"
          >
            Close Camera
          </button>
        )}

        {cameraActive && (
          <div className="w-full flex flex-col items-center">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full h-auto rounded-xl shadow-lg border-4 border-amber-700 mb-4"
            />
            <button
              onClick={captureImage}
              className="bg-green-700 hover:bg-green-800 text-white px-6 py-2 rounded-lg font-medium shadow transition"
            >
              Capture Image
            </button>
          </div>
        )}

        {imageURL && (
          <div className="w-full flex flex-col items-center">
            <img
              src={imageURL}
              alt="Captured preview"
              className="w-full h-auto rounded-xl shadow-lg border-4 border-amber-700 mb-4"
            />
            <button
              onClick={handlePredict}
              disabled={isPredicting}
              className={`${
                isPredicting ? 'bg-gray-500 cursor-not-allowed' : 'bg-green-700 hover:bg-green-800'
              } text-white px-6 py-2 rounded-lg font-medium shadow transition`}
            >
              {isPredicting ? 'Analyzing...' : 'Predict Roast'}
            </button>
          </div>
        )}

        {prediction && (
          <div className="w-full bg-amber-100 rounded-xl p-6 shadow-lg border border-amber-300 mt-6">
            <h2 className="text-2xl font-bold text-amber-900 mb-4">
              Roast Level: <span className="text-amber-700">{prediction.name}</span>
            </h2>
            <div className="bg-white p-4 rounded-lg shadow-inner">
              <h3 className="font-semibold text-amber-800 mb-2">Flavor Characteristics:</h3>
              <p className="text-amber-900">{prediction.description}</p>
            </div>
          </div>
        )}

        {error && (
          <div className="w-full mt-4 p-3 bg-red-100 text-red-700 rounded-lg text-center">
            {error}
          </div>
        )}
      </div>

      {/* Hidden canvas for image capture */}
      <canvas ref={canvasRef} className="hidden" />

      {/* Model preloader */}
      {isModelLoading && (
        <div className="fixed inset-0 bg-white bg-opacity-90 flex items-center justify-center z-50">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-amber-700 mb-4 mx-auto"></div>
            <p className="text-lg font-medium text-amber-900">Loading AI Model...</p>
          </div>
        </div>
      )}

      {/* Prediction preloader */}
      {isPredicting && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-40">
          <div className="bg-white rounded-xl p-8 flex flex-col items-center shadow-lg">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-amber-700 mb-4"></div>
            <p className="text-lg font-medium text-gray-700">Analyzing coffee beans...</p>
          </div>
        </div>
      )}

      <footer className="mt-12 text-center text-amber-700 text-sm">
        <p>Point your camera at coffee beans to analyze their roast level</p>
        <p className="mt-2">Ensure the beans are clearly visible and well-lit</p>
      </footer>
    </div>
  );
}