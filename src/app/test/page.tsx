'use client';
import { useEffect, useRef, useState } from 'react';
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
  },
];

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [prediction, setPrediction] = useState<{ name: string; description: string } | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cameraActive, setCameraActive] = useState(false);

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
  }, []);

  const startCamera = async () => {
    setPrediction(null);
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setCameraActive(true);
      }
    } catch (err) {
      console.error(err);
      setError('Unable to access camera.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setCameraActive(false);
    }
  };

  const captureAndPredict = async () => {
    if (!model || !videoRef.current || !canvasRef.current) return;

    setIsPredicting(true);
    setError(null);

    const video = videoRef.current;
    const canvas = canvasRef.current;

    canvas.width = 256;
    canvas.height = 256;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    try {
      const tensor = tf.tidy(() =>
        tf.browser
          .fromPixels(imageData)
          .toFloat()
          .div(255.0)
          .expandDims()
      );

      const output = model.predict(tensor) as tf.Tensor;
      const data = await output.data();
      tensor.dispose();
      output.dispose();

      const predictedIndex = Array.from(data).indexOf(Math.max(...Array.from(data)));
      setPrediction(classLabels[predictedIndex]);
    } catch (err) {
      console.error(err);
      setError('Prediction failed.');
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="min-h-screen bg-amber-50 flex flex-col items-center p-6">
      <h1 className="text-3xl md:text-4xl font-bold text-amber-900 mb-6 text-center">
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
          <>
            <video ref={videoRef} className="rounded-xl shadow-lg border-4 border-amber-700 mb-4 w-full" />
            <button
              onClick={captureAndPredict}
              disabled={isPredicting}
              className={`${
                isPredicting ? 'bg-gray-500' : 'bg-green-700 hover:bg-green-800'
              } text-white px-6 py-2 rounded-lg font-medium shadow transition`}
            >
              {isPredicting ? 'Analyzing...' : 'Capture & Predict'}
            </button>
            <button
              onClick={stopCamera}
              className="mt-3 text-sm text-red-600 hover:underline"
            >
              Close Camera
            </button>
          </>
        )}

        {/* Hidden canvas for capturing frame */}
        <canvas ref={canvasRef} className="hidden" />

        {prediction && (
          <div className="w-full bg-amber-100 rounded-xl p-6 shadow-lg border border-amber-300 mt-6">
            <h2 className="text-2xl font-bold text-amber-900 mb-4">
              Roast Level: <span className="text-amber-700">{prediction.name}</span>
            </h2>
            <div className="bg-white p-4 rounded-lg shadow-inner">
              <h3 className="font-semibold text-amber-800 mb-2">Ciri-ciri Rasa:</h3>
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
        <p>Arahkan kamera ke biji kopi dan tekan tombol prediksi</p>
        <p className="mt-2">Pastikan gambar terlihat jelas dan cukup pencahayaan</p>
      </footer>
    </div>
  );
}
