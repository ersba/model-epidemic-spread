using Newtonsoft.Json;
using Mars.Interfaces;
using Mars.Interfaces.Environments;
using System;
using System.IO;
using System.Collections.Generic;
using static Tensorflow.Binding;
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
    public class ContactGraphEnvironment : IEnvironment, IModelObject
    {
        private Tensor _edges;

        private Tensor _sfSusceptibility; 
        
        private Tensor _sfInfector;

        private Tensor _edgeAttributes;
        
        public Tensor LamdaGammaIntegrals { get; set; }
        
        public ContactGraphEnvironment(int agentCount)
        {
            InitEdgesWithCsv(agentCount); 
            _sfSusceptibility = tf.constant(new float[] {0.35f, 0.69f, 1.03f, 1.03f, 1.03f, 1.03f, 1.27f, 1.52f});
            _sfInfector = tf.constant(new float[] {0.0f, 0.33f, 0.72f, 0.0f, 0.0f});
            _edgeAttributes = tf.tile(tf.constant(1f), tf.constant(new int[] { (int)_edges.shape[1] }));
            
        }
        
        private void InitEdgesWithCsv(int limit)
        {
            var firstPart = new List<float>();
            var secondPart = new List<float>();
            
            foreach (var line in File.ReadAllLines("Resources/contact_edges.csv"))
            {
                var splitLine = line.Split(',');
                float firstNumber = float.Parse(splitLine[0]);
                float secondNumber = float.Parse(splitLine[1]);

                if (firstNumber < limit && secondNumber < limit)
                {
                    firstPart.Add(firstNumber);
                    secondPart.Add(secondNumber);
                }
            }
            
            int length = firstPart.Count;
            
            float[,] tensorArray = new float[2, length];
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
        
    }
}