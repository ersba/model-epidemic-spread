using System;
using System.IO;
using System.Linq;
using Tensorflow;

using Tensorflow.Keras.Models;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Losses;


namespace EpidemicSpread
{
    public class SimpleCalibNn
    {
        private Sequential _model;
        
        private NDArray _features;
        
        private NDArray _labels;
        
        public SimpleCalibNn()
        {
            LoadData();
            InitModel();
            CompileModel();
        }
        
        public void Train(int epochs = 10)
        {
            _model.fit(_features, _labels, batch_size: 1, epochs: 1, verbose: 1);
            _model.save("Resources/model.json");
            _model.save_weights("Resources/model.h5");
        }
        private void LoadData()
        {
            var filePath = "Resources/training.csv"; 
            var lines = File.ReadAllLines(filePath).Skip(1).ToArray();
            var featureData = lines.Select(line => float.Parse(line.Split(',')[0])).ToArray();
            var labelData = lines.Select(line => float.Parse(line.Split(',')[1])).ToArray();

            _features = np.array(featureData).reshape(new Shape(-1, 1)); // Stellen Sie sicher, dass die Dimensionen stimmen
            _labels = np.array(labelData).reshape(new Shape(-1, 1)); // Stellen Sie sicher, dass die Dimensionen stimmen

        }
        
        private void InitModel()
        {
            string modelPath = "Resources/model.json";
            string weightPath = "Resources/model.h5";
            
            if (File.Exists(modelPath) && File.Exists(weightPath))
            {
                var model = keras.models.load_model(modelPath);
                model.load_weights(weightPath);
                _model = (Sequential)model;
            }
            else
            {
                _model = keras.Sequential();
                _model.add(keras.layers.Dense(32, activation: "relu", input_shape: new Shape(1)));
                _model.add(keras.layers.Dense(64, activation: "relu"));
                _model.add(keras.layers.Dense(1, activation: "sigmoid"));
            }
        }
        private void CompileModel()
        {
            // _model.compile(optimizer: keras.optimizers.Adam(), loss: keras.losses.MeanAbsoluteError(), metrics: new[] {"accuracy"});
            _model.compile(optimizer: keras.optimizers.Adam(), loss: new CustomLoss(), metrics: new[] {"accuracy"});
        }
    }
    class CustomLoss : ILossFunc
    {
        public Tensor Call(Tensor yTrue, Tensor yPred, Tensor sampleWeight = null)
        {
            
            var predictedDeaths = Program.Deaths;
            // Program.Dispose();
            var loss = tf.reduce_mean(tf.square(yTrue - predictedDeaths));
            return loss;
        }

        public string Reduction { get; }
        public string Name { get; }
    }
}