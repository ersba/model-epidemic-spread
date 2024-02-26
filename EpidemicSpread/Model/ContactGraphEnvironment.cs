﻿using Newtonsoft.Json;
using Mars.Interfaces;
using Mars.Interfaces.Environments;
using System;
using System.IO;
using System.Collections.Generic;
using static Tensorflow.Binding;
using MathNet.Numerics.Distributions;
using static Tensorflow.KerasApi;
using System.Linq;
using Mars.Components.Layers;
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

        private Tensor _lamdaGammaIntegrals;

        private Tensor _exposedToday;
        
        private int[] _arrayExposedToday;
        
        public ContactGraphEnvironment()
        {
            InitEdgesWithCsv(Params.AgentCount); 
            SetLamdaGammaIntegrals(5.15, 2.14, Params.Steps);
            _arrayExposedToday = new int[Params.AgentCount];
        }

        public bool Interact(int index)
        {
            return _arrayExposedToday[index] == 1;
        }
        
        protected override Tensor Message(Tensor sourceFeature, Tensor targetFeature, int currentTick)
        {   
            return Lamda(sourceFeature, targetFeature, currentTick);
        }

        public Tensor Forward(Tensor nodeFeatures, int currentTick)
        {
            var lamda = tf.reshape(Propagate(_edges, nodeFeatures, currentTick), new Shape(-1,1));
            var probabilityNotInfected = tf.exp(-lamda);
            var p =  tf.concat(new [] { 1 - probabilityNotInfected, probabilityNotInfected }, axis: 1);
            var potentiallyExposedToday = GumbelSoftmax.Execute(p)[Slice.All,0];
            UpdateExposedToday(potentiallyExposedToday, nodeFeatures);
            _arrayExposedToday = tf.cast(_exposedToday, TF_DataType.TF_INT32).numpy().ToArray<int>();
            return tf.expand_dims(_exposedToday, axis: 1);
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
            
            var forwardEdges = tf.constant(tensorArray);
            var backwardEdges = tf.stack(new [] 
            {
                forwardEdges[1],
                forwardEdges[0]
            }, axis: 0);
            
            _edges = tf.concat(new [] 
            {
                forwardEdges, 
                backwardEdges
            }, axis: 1);
        }

        private Tensor Lamda(Tensor sourceFeature, Tensor targetFeature, int currentTick)
        {
            var targetAgeGroup = tf.gather(targetFeature, tf.constant(0), axis: 1);
            var targetSusceptibility = tf.gather(Params.Susceptibility, tf.cast(targetAgeGroup, TF_DataType.TF_INT32));
            var sourceStage = tf.gather(sourceFeature, tf.constant(1), axis: 1);
            var sourceInfector = tf.gather(Params.Infector, tf.cast(sourceStage,TF_DataType.TF_INT32));
            var bN = Params.EdgeAttribute;
            var integrals = tf.cast(tf.zeros_like(sourceStage), TF_DataType.TF_FLOAT);
            var sourceInfectedIndex = tf.cast(tf.gather(sourceFeature, tf.constant(2), axis: 1), dtype: TF_DataType.TF_BOOL);
            // var sourceInfectedTime = tf.gather(tf.boolean_mask(sourceFeature, sourceInfectedIndex, axis: 0), tf.constant(3), axis: 1);
            var sourceInfectedTime = tf.gather(sourceFeature, tf.constant(3), axis: 1);
            var tick = tf.ones_like(sourceInfectedTime) * currentTick;
            sourceInfectedTime = tf.abs(tick - sourceInfectedTime);
            integrals = tf.where(sourceInfectedIndex, tf.gather(_lamdaGammaIntegrals, tf.cast(sourceInfectedTime, TF_DataType.TF_INT32)), integrals);
            var meanInteractions = tf.gather(targetFeature, tf.constant(4), axis: 1);
            // tf.print(tf.shape(targetSusceptibility));
            // tf.print(tf.shape(sourceInfector));
            // tf.print(tf.shape(integrals));
            // tf.print(tf.shape(meanInteractions));
            // Console.WriteLine("---");
            var result = LearnableParams.Instance.R0Value * targetSusceptibility * sourceInfector * bN * integrals / meanInteractions;
            // tf.print(tf.shape(result));
            // tf.print(tf.shape(tf.reshape(result, new Shape(-1,1))));
            // return tf.reshape(result, new Shape(-1,1));
            return result;
        }
        
        // This multiplication ensures that only agents are marked as newly exposed who were both susceptible and
        // exposed to a possible source of infection
        private void UpdateExposedToday(Tensor potentiallyExposed, Tensor nodeFeatures)
        {
            var susceptibleMask = tf.equal(tf.gather(nodeFeatures, tf.constant(1), axis: 1), tf.constant((int)Stage.Susceptible, TF_DataType.TF_FLOAT));
            // tf.print(tf.shape(susceptibleMask));
            _exposedToday = tf.cast(susceptibleMask, TF_DataType.TF_INT32) * potentiallyExposed;
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