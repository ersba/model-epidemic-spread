using Newtonsoft.Json;
using Mars.Interfaces;
using Mars.Interfaces.Environments;
using System;
using System.IO;
using System.Collections.Generic;
using static Tensorflow.Binding;
using MathNet.Numerics.Distributions;
using static Tensorflow.KerasApi;
using System.Linq;
using NetTopologySuite.Planargraph;
using ServiceStack;
using ServiceStack.Text;
using Tensorflow;
using Tensorflow.Keras.Utils;
using Tensorflow.NumPy;


namespace EpidemicSpread.Model
{
    public class ContactGraphEnvironment : TensorGraphEnvironment, IEnvironment, IModelObject
    {
        
        private Tensor _edges;

        private Tensor _edgeAttributes;

        private Tensor _nodeFeatures;

        private Tensor _sfSusceptibility; 
        
        private Tensor _sfInfector;

        private Tensor _lamdaGammaIntegrals;
        
        public ContactGraphEnvironment(int agentCount, int steps)
        {
            InitEdgesWithCsv(agentCount); 
            SetLamdaGammaIntegrals(5.15, 2.14, steps);
            _sfSusceptibility = tf.constant(new float[] {0.35f, 0.69f, 1.03f, 1.03f, 1.03f, 1.03f, 1.27f, 1.52f});
            _sfInfector = tf.constant(new float[] {0.0f, 0.33f, 0.72f, 0.0f, 0.0f});
            // _edgeAttributes = tf.tile(tf.constant(1f), tf.constant(new int[] { (int)_edges.shape[1] }));
            _edgeAttributes = tf.constant(1f);
        }
        
        protected override Tensor Message(Tensor sourceFeature, Tensor targetFeature, params object[] args)
        {   
            return Lamda(sourceFeature, targetFeature, (int)args[0]);
        }

        public Tensor Forward(Tensor nodeFeatures, int currentTick)
        {
            return Propagate(_edges, nodeFeatures, currentTick);
        }
        
        private void InitEdgesWithCsv(int limit)
        {
            var firstPart = new List<int>();
            var secondPart = new List<int>();
            
            foreach (var line in File.ReadAllLines("Resources/contact_edges.csv"))
            {
                var splitLine = line.Split(',');
                int firstNumber = int.Parse(splitLine[0]);
                int secondNumber = int.Parse(splitLine[1]);

                if (firstNumber < limit && secondNumber < limit)
                {
                    firstPart.Add(firstNumber);
                    secondPart.Add(secondNumber);
                }
            }
            
            int length = firstPart.Count;
            
            int[,] tensorArray = new int[2, length];
            for (int i = 0; i < length; i++)
            {
                tensorArray[0, i] = firstPart[i];
                tensorArray[1, i] = secondPart[i];
            }
            
            var forwardEdges = tf.Variable(tensorArray);
            var backwardEdges = tf.stack(new [] 
            {
                forwardEdges[1],
                forwardEdges[0]
            }, axis: 0);
            
            _edges = tf.Variable(tf.concat(new [] 
            {
                forwardEdges, 
                backwardEdges
            }, axis: 1));
        }

        private Tensor Lamda(Tensor sourceFeature, Tensor targetFeature, int step)
        {
            // tf.print(tf.shape(sourceFeature));
            var targetAgeGroup = tf.gather(targetFeature, tf.constant(0), axis: 1);
            var targetSusceptibility = tf.gather(_sfSusceptibility, targetAgeGroup);
            var sourceStage = tf.gather(sourceFeature, tf.constant(1), axis: 1);
            var sourceInfector = tf.gather(_sfInfector, sourceStage);
            var bN = _edgeAttributes;
            var integrals = tf.cast(tf.zeros_like(sourceStage), TF_DataType.TF_FLOAT);
            var sourceInfectedIndex = tf.cast(tf.gather(sourceFeature, tf.constant(2), axis: 1), dtype: TF_DataType.TF_BOOL);
            // var sourceInfectedTime = tf.gather(tf.boolean_mask(sourceFeature, sourceInfectedIndex, axis: 0), tf.constant(3), axis: 1);
            var sourceInfectedTime = tf.gather(sourceFeature, tf.constant(3), axis: 1);
            tf.print(sourceInfectedTime);
            var tick = tf.ones_like(sourceInfectedTime) * step;
            sourceInfectedTime = tick - sourceInfectedTime;
            // tf.print(tf.reduce_sum(tf.cast(sourceInfectedIndex, TF_DataType.TF_INT32)));
            var lami = tf.gather(_lamdaGammaIntegrals, sourceInfectedTime);
            integrals = tf.where(sourceInfectedIndex, lami, integrals);
            tf.print(integrals);
            // var meanInteractions = tf.gather(targetFeature, tf.constant(4), axis: 1);
            // tf.print(tf.shape(targetSusceptibility));
            // tf.print(tf.shape(sourceInfector));
            // tf.print(tf.shape(integrals));
            // tf.print(tf.shape(meanInteractions));
            Console.WriteLine("---");
            return tf.constant(1);
        }
        
        private void SetLamdaGammaIntegrals(double scale, double rate, int steps)
        {
            double b = rate * rate / scale;
            double a = scale / b;
            var res = new List<float>();

            for (int t = 1; t <= steps + 10; t++)
            {
                double cdfAtTimeT = Gamma.CDF(a, b, t);
                double cdfAtTimeTMinusOne = Gamma.CDF(a, b, t - 1);
                res.Add((float)(cdfAtTimeT - cdfAtTimeTMinusOne));
            }
            _lamdaGammaIntegrals = tf.constant(res.ToArray(), TF_DataType.TF_FLOAT);
        }
        
    }
}