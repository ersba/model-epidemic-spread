using System;
using System.IO;
using System.Linq;
using EpidemicSpread.Model;
using ServiceStack;
using Tensorflow;
using Tensorflow.Keras;
using Tensorflow.Keras.ArgsDefinition;
using Tensorflow.Keras.Models;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Layers;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Metrics;
using Tensorflow.Keras.Optimizers;
using Tensorflow.Operations.Activation;

namespace EpidemicSpread
{
    public class SimpleCalibNn
    {
        private Sequential Model;
        
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
            
             Model.fit(_features, _labels, batch_size: 1, epochs: epochs, verbose: 1);
            Model.save(_modelPath, save_format:"tf");
             // _model.save_weights(weightsPath);
        }

        public void CustomTrain(int epochs = 10)
        {
            var optimizer = new Adam();
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                using (var tape = tf.GradientTape())
                {
                    // Berechnen Sie die Vorhersagen
                    // var predictions = ((Tensor)_model.predict(_features))[0, 1];
                    var predictions = (Tensor)Model.predict(_features);
                    // Berechnen Sie den Verlust mit Ihrer benutzerdefinierten Funktion
                    // var loss = CustomLoss(_labels, predictions) + tf.stop_gradient(predictions);
                    // var loss = CustomLoss(_labels, predictions) + predictions;

                    var loss = CustomLoss(_labels, predictions);

                    var gradients = tape.gradient(loss, Model.TrainableVariables);
                    tf.print(gradients[5]);
                    optimizer.apply_gradients(zip(gradients, Model.TrainableVariables));
                    
                    // Console.WriteLine($"Epoche: {epoch + 1}, Verlust: {loss.numpy()}, Vorhersage: {predictions}, Gradient: {gradients[1]}");

                    Console.WriteLine($"Epoche: {epoch + 1}, Verlust: {loss.numpy()}, Vorhersage: {predictions}");
                }
            }
        }

        private Tensor CustomLoss(Tensor target, Tensor predictions)
        {
            var lowerBounds = tf.constant(new float[] {1.0f, 0.001f, 0.01f, 2.0f, 4.0f});
            var upperBounds = tf.constant(new float[] {9.0f, 0.9f, 0.9f, 6.0f, 7.0f});
            var boundedPred = lowerBounds + (upperBounds - lowerBounds) * predictions;

            LearnableParams learnableParams = LearnableParams.Instance;
            
            Console.Write("Params:");
            tf.print(predictions);
            // var softSample = tf.constant(boundedPred.numpy());
            // var hardSample = tf.cast(softSample,TF_DataType.TF_INT32);
            // learnableParams.MortalityRate = predictions[0, 2];
            learnableParams.InitialInfectionRate = predictions[0, 1];
            // learnableParams.InfectedToRecoveredTime = tf.stop_gradient(tf.cast(tf.constant(boundedPred[0,4].numpy(), 
            //     TF_DataType.TF_INT32), TF_DataType.TF_FLOAT) - boundedPred[0,4]) + boundedPred[0,4];
            // tf.print(learnableParams.InfectedToRecoveredTime);
            var predictedDeaths = Program.EpidemicSpreadSimulation();



            // var predictedDeaths = SimpleSimulation.Execute();
            Console.Write("Deaths: ");
            tf.print(predictedDeaths);
            
            // learnableParams.R0Value = tf.constant(5.18, dtype: TF_DataType.TF_FLOAT);
            // learnableParams.InitialInfectionRate = tf.constant(0.5, dtype: TF_DataType.TF_FLOAT);
            // learnableParams.MortalityRate = tf.constant(0.9, dtype: TF_DataType.TF_FLOAT);
            // learnableParams.ExposedToInfectedTime = tf.constant(3, dtype: TF_DataType.TF_FLOAT);
            // learnableParams.InfectedToRecoveredTime = tf.constant(5, dtype: TF_DataType.TF_FLOAT);
            // tf.print(yTrue);
            // tf.print(predictedDeaths);
            var loss = tf.reduce_mean(tf.square(target - predictedDeaths));
            
            return loss;
        }

        private Tensor CustomLossGumbel(Tensor target, Tensor prediction)
        {
            tf.print(prediction);
            var ones = tf.ones(new Shape(1000, 1));
            Tensor predColumn = ones * prediction;
            Tensor oneMinusPredColumn = ones * (1 - prediction);
            Tensor pTiled = tf.concat(new [] {predColumn, oneMinusPredColumn}, axis: 1);
            tf.print(tf.shape(pTiled));
            var infected = tf.reduce_sum(tf.cast(GumbelSoftmax.Execute(pTiled)[Slice.All, 0], dtype: TF_DataType.TF_FLOAT));
            tf.print(infected);
            return tf.reduce_mean(tf.square(target - infected));
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
                Model = (Sequential)model;
                // _model.load_weights(weightsPath);
                // _model.summary();
            }
            else
            {
                Model = keras.Sequential();
                Model.add(keras.layers.Dense(units: 16, activation: null, input_shape: new Shape(1)));
                Model.add(keras.layers.LeakyReLU());
                Model.add(keras.layers.Dense(16));
                Model.add(keras.layers.LeakyReLU());
                Model.add(keras.layers.Dense(5, activation: "sigmoid"));
                
                // var inputs = keras.Input(shape: new Shape(1));
                // var x = new Dense(new DenseArgs
                // {Units = 32, Activation = keras.activations.Linear}).Apply(inputs);
                // x = new Dense(new DenseArgs { Units = 64}).Apply(x);
                // x = new Dense(new DenseArgs { Units = 64}).Apply(x);
                // x = new Dense(new DenseArgs { Units = 64}).Apply(x);  // Optional: LeakyReLU hinzufügen, wenn benötigt
                // var outputs = new Dense(new DenseArgs {Units = 5, Activation = keras.activations.Sigmoid}).Apply(x);
                //
                // _model = new Functional(inputs, outputs);
            }
            Model.compile(optimizer: keras.optimizers.Adam(), loss: new CustomLoss());
        }
    }
    
    class CustomLoss : ILossFunc
    {
        public Tensor Call(Tensor yTrue, Tensor yPred, Tensor sampleWeight = null)
        {
            
            
            var lowerBounds = tf.constant(new float[] {1.0f, 0.001f, 0.01f, 2.0f, 4.0f});
            var upperBounds = tf.constant(new float[] {9.0f, 0.9f, 0.9f, 6.0f, 7.0f});
            var boundedPred = lowerBounds + (upperBounds - lowerBounds) * yPred;
            
            LearnableParams learnableParams = LearnableParams.Instance;
            
            Console.Write("Params:");
            tf.print(boundedPred);
            // var softSample = tf.constant(boundedPred.numpy());
            // var hardSample = tf.cast(softSample,TF_DataType.TF_INT32);
            // tf.print(hardSample);
            // tf.print(learnableParams.InitialInfectionRate);
            learnableParams.MortalityRate = boundedPred[0, 2];
            // learnableParams.InitialInfectionRate = boundedPred[0, 1];
            // learnableParams.InfectedToRecoveredTime = tf.cast(tf.equal(boundedPred[0, 4], tf.reduce_max(boundedPred[0, 4], axis: 1, keepdims: true)),TF_DataType.TF_INT32);
            // tf.print(learnableParams.InfectedToRecoveredTime);
            // var predictedDeaths = Program.EpidemicSpreadSimulation();
            var predictedDeaths = tf.constant(500) * learnableParams.MortalityRate + tf.stop_gradient(boundedPred[0, 2]);
            
            
            
            // var predictedDeaths = SimpleSimulation.Execute();
            Console.Write("Deaths: ");
            tf.print(predictedDeaths);
            
            // learnableParams.R0Value = tf.constant(5.18, dtype: TF_DataType.TF_FLOAT);
            // learnableParams.InitialInfectionRate = tf.constant(0.5, dtype: TF_DataType.TF_FLOAT);
            // learnableParams.MortalityRate = tf.constant(0.9, dtype: TF_DataType.TF_FLOAT);
            // learnableParams.ExposedToInfectedTime = tf.constant(3, dtype: TF_DataType.TF_FLOAT);
            // learnableParams.InfectedToRecoveredTime = tf.constant(5, dtype: TF_DataType.TF_FLOAT);
            // tf.print(yTrue);
            // tf.print(predictedDeaths);
            var loss = tf.reduce_mean(tf.square(yTrue - predictedDeaths));
            
            return loss;
        }

        public string Reduction { get; }
        public string Name { get; }
    }
}