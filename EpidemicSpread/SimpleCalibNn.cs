using System;
using System.IO;
using System.Linq;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.Models;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Metrics;
using Tensorflow.Operations.Activation;

namespace EpidemicSpread
{
    public class SimpleCalibNn
    {
        private Sequential _model;
        
        private NDArray _features;
        
        private NDArray _labels;
        
        private string _modelPath;

        public Tensor Predicted;
        
        public SimpleCalibNn()
        {
            string projectDirectory = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;
            _modelPath = Path.Combine(projectDirectory, "simple_calibnn");
            LoadData();
            InitModel();
            
        }
        
        public void Train(int epochs = 10)
        {
            // string projectDirectory = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;
            // string modelPath = Path.Combine(projectDirectory, "modelnn");
            // string weightsPath = Path.Combine(projectDirectory, "model.h5");
            
             _model.fit(_features, _labels, batch_size: 1, epochs: epochs, verbose: 1);
            _model.save(_modelPath, save_format:"tf");
             // _model.save_weights(weightsPath);
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
            string projectDirectory = Directory.GetParent(Environment.CurrentDirectory).Parent.Parent.FullName;
            string weightsPath = Path.Combine(projectDirectory, "model.h5");
            
            
            // string modelPath = "model";
            // string weightsPath = "model.h5";
            // Console.WriteLine(Directory.GetCurrentDirectory());
            // var model = keras.models.load_model(modelPath);
            Console.WriteLine(_modelPath);
            Console.WriteLine(Directory.Exists(_modelPath));
            if (Directory.Exists(_modelPath))
            {
                var model = keras.models.load_model(_modelPath);
                // model.load_weights(weightsPath);
                _model = (Sequential)model;
                // _model.load_weights(weightsPath);
                // _model.summary();
            }
            else
            {
                _model = keras.Sequential();
                _model.add(keras.layers.Dense(units: 32, activation: null, input_shape: new Shape(1)));
                _model.add(keras.layers.LeakyReLU(alpha: 0.01f));
                _model.add(keras.layers.Dense(64));
                _model.add(keras.layers.LeakyReLU(alpha: 0.01f));
                _model.add(keras.layers.Dense(5, activation: "sigmoid"));
                // _model.load_weights(weightsPath);
            }
            _model.compile(optimizer: keras.optimizers.Adam(), loss: new CustomLoss());
        }
    }
    
    class CustomLoss : ILossFunc
    {
        public Tensor Call(Tensor yTrue, Tensor yPred, Tensor sampleWeight = null)
        {
            var lowerBounds = tf.constant(new float[] {1.0f, 0.3f, 0.001f, 2.0f, 4.0f});
            var upperBounds = tf.constant(new float[] {9.0f, 0.9f, 0.9f, 6.0f, 7.0f});
            var boundedPred = lowerBounds + (upperBounds - lowerBounds) * yPred;
            
            LearnableParams learnableParams = LearnableParams.Instance;
            
            var softSample = tf.constant(boundedPred.numpy());
            var hardSample = tf.cast(softSample,TF_DataType.TF_INT32);
            // tf.print(hardSample);
            // learnableParams.MortalityRate = tf.cast(tf.stop_gradient(hardSample - softSample)+ yPred[0, 0], TF_DataType.TF_FLOAT);
            learnableParams.InitialInfectionRate = boundedPred[0, 1];
            // learnableParams.InfectedToRecoveredTime = tf.cast(yPred[0, 0], TF_DataType.DtInt32Ref);
            Console.Write("Parameter: ");
            tf.print(yPred[0,1]);
            var predictedDeaths = Program.EpidemicSpreadSimulation();
            Console.Write("Deaths: ");
            tf.print(predictedDeaths);
            
            learnableParams.R0Value = tf.constant(5.18, dtype: TF_DataType.TF_FLOAT);
            learnableParams.InitialInfectionRate = tf.constant(0.5, dtype: TF_DataType.TF_FLOAT);
            learnableParams.MortalityRate = tf.constant(0.9, dtype: TF_DataType.TF_FLOAT);
            learnableParams.ExposedToInfectedTime = tf.constant(3, dtype: TF_DataType.TF_FLOAT);
            learnableParams.InfectedToRecoveredTime = tf.constant(5, dtype: TF_DataType.TF_FLOAT);
            
            var loss = tf.square(yTrue - predictedDeaths);
            
            return loss;
        }

        public string Reduction { get; }
        public string Name { get; }
    }
}