﻿using System.Linq;
using System;
using System.Collections.Generic;
using Mars.Common.Core.Collections;
using Mars.Common.Core.Random;
using Mars.Components.Environments;
using Mars.Components.Environments.Cartesian;
using Mars.Components.Layers;
using Mars.Core.Data;
using Mars.Interfaces;
using Mars.Interfaces.Annotations;
using Mars.Interfaces.Data;
using Mars.Components.Services;
using Mars.Interfaces.Environments;
using Mars.Interfaces.Layers;
using Mars.Numerics;
using ServiceStack;
using MathNet.Numerics.Distributions;
using MongoDB.Driver;
using Tensorflow;
using Tensorflow.Keras.Metrics;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.NumPy;

namespace EpidemicSpread.Model

{
    public class InfectionLayer : AbstractLayer, ISteppedActiveLayer
    {
        [PropertyDescription]
        public int AgentCount { get; set; }
        
        [PropertyDescription]
        public int Steps { get; set; }
        
        public ContactGraphEnvironment ContactEnvironment { get; private set; }
        
        public ResourceVariable AgeGroups { get; private set; }
        
        public ResourceVariable Stages { get; private set; }
        
        public ResourceVariable MeanInteractions { get; private set; }
        
        public IAgentManager AgentManager { get; private set; }

        private LearnableParams _learnableParams;

        private const int ChildUpperIndex = 1;

        private const int AdultUpperIndex = 6;
        
        private const float Susceptible = 0f;
        private const float Exposed = 1f; 
        private const float Infected = 2f;
        private const float Recovered = 3f;
        private const float Mortality = 4f;

        private ResourceVariable _infectedTime;

        private ResourceVariable _nextStageTimes;
        
        

        private static readonly float[] Mu = { 2, 4, 3 };
        
        

        public override bool InitLayer(LayerInitData layerInitData, RegisterAgent registerAgentHandle,
            UnregisterAgent unregisterAgentHandle)
        {
            var initiated = base.InitLayer(layerInitData, registerAgentHandle, unregisterAgentHandle);
            ContactEnvironment = new ContactGraphEnvironment(AgentCount);
            AgeGroups = tf.Variable(tf.zeros(new Shape(AgentCount, 1), TF_DataType.DtInt32Ref));
            AgentManager = layerInitData.Container.Resolve<IAgentManager>();
            AgentManager.Spawn<Host, InfectionLayer>().ToList();
            _learnableParams = LearnableParams.Instance;
            InitStages();
            InitMeanInteractions();
            InitTimeVariables();
            SetLamdaGammaIntegrals(5.5,2.14);
            
            
            return initiated;
        }

        public void Tick()
        {
            
        }

        public void PreTick()
        { 
            
        }

        public void PostTick()
        {
            
        }
        
        private void InitStages()
        {
            var probabilityInfected = _learnableParams.InitialInfectionRate / tf.constant(100) * tf.ones(new Shape(AgentCount, 1));
            var p = tf.Variable(tf.concat(new [] { probabilityInfected, 1 - probabilityInfected }, axis: 1));
            Stages = tf.Variable(GumbelSoftmax.Execute(p)[Slice.All,0], dtype: TF_DataType.TF_FLOAT);
            // tf.print(Stages[0]);
        }

        private void InitMeanInteractions()
        {
            MeanInteractions = tf.Variable(tf.zeros(new Shape(AgentCount, 1)));
            var childAgents = tf.less_equal(AgeGroups, ChildUpperIndex);
            var adultAgents = tf.logical_and(tf.greater(AgeGroups, ChildUpperIndex), tf.less_equal(AgeGroups, AdultUpperIndex));
            var elderlyAgents = tf.greater(AgeGroups, AdultUpperIndex);
            MeanInteractions = tf.Variable(MeanInteractions.assign(tf.where(
                childAgents, tf.fill(tf.shape(MeanInteractions), Mu[0]),
                MeanInteractions)));
            MeanInteractions = tf.Variable(MeanInteractions.assign(tf.where(
                adultAgents, tf.fill(tf.shape(MeanInteractions), Mu[1]),
                MeanInteractions)));
            MeanInteractions = tf.Variable(MeanInteractions.assign(tf.where(
                elderlyAgents, tf.fill(tf.shape(MeanInteractions), Mu[2]),
                MeanInteractions)));
        }
        
        private void InitTimeVariables()
        {
            _infectedTime = tf.Variable((Steps + 1) * tf.ones(new Shape(AgentCount, 1)));
            _nextStageTimes = tf.Variable((Steps + 1) * tf.ones(new Shape(AgentCount, 1)));
            
            var exposedCondition = tf.equal(Stages, tf.Variable(Exposed, dtype: TF_DataType.TF_FLOAT));
            var infectedCondition = tf.equal(Stages, tf.Variable(Infected, dtype: TF_DataType.TF_FLOAT));
            
            _infectedTime = tf.Variable(tf.where(exposedCondition, tf.fill(tf.shape(_infectedTime), tf.Variable(0, dtype:TF_DataType.TF_FLOAT)), _infectedTime));
            _infectedTime = tf.Variable(tf.where(infectedCondition, tf.fill(tf.shape(_infectedTime), (tf.Variable(-1, dtype:TF_DataType.TF_FLOAT) * _learnableParams.ExposedToInfectedTime) + tf.Variable(1, dtype:TF_DataType.TF_FLOAT)), _infectedTime));
            
            _nextStageTimes = tf.Variable(tf.where(exposedCondition, tf.fill(tf.shape(_nextStageTimes), (_learnableParams.ExposedToInfectedTime) + tf.Variable(1, dtype:TF_DataType.TF_FLOAT)), _nextStageTimes));
            _nextStageTimes = tf.Variable(tf.where(infectedCondition, tf.fill(tf.shape(_nextStageTimes), (_learnableParams.InfectedToRecoveredTime) + tf.Variable(1, dtype:TF_DataType.TF_FLOAT)), _nextStageTimes));
        }

        private void SetLamdaGammaIntegrals(double scale, double rate)
        {
            double b = rate * rate / scale;
            double a = scale / b;
            var res = new List<float>();

            for (int t = 1; t <= Steps + 10; t++)
            {
                double cdfAtTimeT = Gamma.CDF(a, b, t);
                double cdfAtTimeTMinusOne = Gamma.CDF(a, b, t - 1);
                res.Add((float)(cdfAtTimeT - cdfAtTimeTMinusOne));
            }
            ContactEnvironment.LamdaGammaIntegrals = tf.constant(res.ToArray(), TF_DataType.TF_FLOAT);
        }
    }
}