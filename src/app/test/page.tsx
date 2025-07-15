'use client';
import { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import { Camera } from 'react-camera-pro';

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
  const cameraRef = useRef<any>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [image, setImage] = useState<string | null>(null);
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
        setError('Failed to load model.');
      } finally {
        setIsModelLoading(false);
      }
    };
    loadModel();
  }, []);

  const activateCamera = () => {
    setCameraActive(true);
    setImage(null);
    setPrediction(null);
    setError(null);
  };

  const takeAndPredict = async () => {
    if (!cameraRef.current || !model) return;

    setIsPredicting(true);
    setPrediction(null);
    setError(null);

    try {
      const photo = cameraRef.current.takePhoto();
      setImage(photo);
      setCameraActive(false);

      const img = new Image();
      img.src = photo;
      img.crossOrigin = 'anonymous';

      img.onload = async () => {
        try {
          const tensor = tf.tidy(() =>
            tf.browser
              .fromPixels(img)
              .resizeBilinear([256, 256])
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
    } catch (err) {
      console.error(err);
      setError('Failed to capture photo.');
      setIsPredicting(false);
    }
  };

  return (
    <div className="min-h-screen bg-amber-50 flex flex-col items-center p-6">
      <h1 className="text-3xl md:text-4xl font-bold text-amber-900 mb-6 text-center">
        Coffee Roast Predictor ☕
      </h1>

      <div className="w-full max-w-lg flex flex-col items-center">
        <div className="w-full aspect-video bg-black rounded-xl overflow-hidden shadow-lg mb-4">
          {cameraActive ? (
            <Camera
              ref={cameraRef}
              aspectRatio={1}
              errorMessages={{
                noCameraAccessible: undefined,
                permissionDenied: undefined,
                switchCamera: undefined,
              }}
            />
          ) : (
            <div className="w-full h-full bg-gray-800 flex items-center justify-center text-white">
              {image ? (
                <img
                  src={image}
                  alt="Captured"
                  className="w-full h-full object-cover"
                />
              ) : (
                <span>Camera not active</span>
              )}
            </div>
          )}
        </div>

        <div className="flex gap-4">
          {!cameraActive && !image && (
            <button
              onClick={activateCamera}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium shadow transition"
            >
              Open Camera
            </button>
          )}

          {cameraActive && (
            <button
              onClick={takeAndPredict}
              disabled={isPredicting}
              className={`${
                isPredicting ? 'bg-gray-500 cursor-not-allowed' : 'bg-green-700 hover:bg-green-800'
              } text-white px-6 py-2 rounded-lg font-medium shadow transition`}
            >
              Take Picture
            </button>
          )}

          {image && !cameraActive && (
            <button
              onClick={activateCamera}
              className="bg-amber-600 hover:bg-amber-700 text-white px-6 py-2 rounded-lg font-medium shadow transition"
            >
              Retake Picture
            </button>
          )}
        </div>

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

      {/* Prediction loader */}
      {isPredicting && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-40">
          <div className="bg-white rounded-xl p-8 flex flex-col items-center shadow-lg">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-amber-700 mb-4"></div>
            <p className="text-lg font-medium text-gray-700">Analyzing coffee beans...</p>
          </div>
        </div>
      )}

      <footer className="mt-12 text-center text-amber-700 text-sm">
        <p>Arahkan kamera ke biji kopi lalu tekan tombol untuk memprediksi tingkat sangrai</p>
        <p className="mt-2">Pastikan pencahayaan cukup dan gambar jelas</p>
      </footer>
    </div>
  );
}