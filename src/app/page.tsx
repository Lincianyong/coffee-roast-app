'use client';
import { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

const classLabels = [
  {
    name: 'Dark',
    description: 'Bold, smoky flavor with low acidity and visible oils on the beans. Often has chocolate or caramel notes.',
  },
  {
    name: 'Green',
    description: 'Unroasted beans with grassy, herbal flavors. Higher acidity and caffeine content before roasting.',
  },
  {
    name: 'Light',
    description: 'Light brown color, no oil on beans. Bright acidity with floral/fruity notes and toasted grain flavors.',
  },
  {
    name: 'Medium',
    description: 'Balanced flavor with medium acidity and body. Shows caramel sweetness with nutty/chocolate undertones.',
  },
];

export default function Home() {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [prediction, setPrediction] = useState<{ name: string; description: string } | null>(null);
  const [imageURL, setImageURL] = useState<string | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

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

  const handleUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;
    const file = files[0];
    const url = URL.createObjectURL(file);
    setImageURL(url);
    setPrediction(null);
    setError(null);
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
        <label
          htmlFor="file-upload"
          className="cursor-pointer bg-amber-700 hover:bg-amber-800 text-white font-medium px-6 py-3 rounded-lg shadow mb-6"
        >
          Upload Image
        </label>
        <input
          id="file-upload"
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleUpload}
          className="hidden"
        />

        {imageURL && (
          <div className="w-full flex flex-col items-center">
            <img
              src={imageURL}
              alt="Uploaded preview"
              className="w-full h-auto rounded-xl shadow-lg border-4 border-amber-700 mb-4"
            />
            <button
              onClick={handlePredict}
              disabled={isPredicting || !model}
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

      {isPredicting && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl p-8 flex flex-col items-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-amber-700 mb-4"></div>
            <p className="text-lg font-medium text-gray-700">Analyzing coffee beans...</p>
          </div>
        </div>
      )}

      <footer className="mt-12 text-center text-amber-700 text-sm">
        <p>Upload an image of coffee beans to analyze their roast level</p>
        <p className="mt-2">Ensure the beans are clearly visible and well-lit</p>
      </footer>
    </div>
  );
}
