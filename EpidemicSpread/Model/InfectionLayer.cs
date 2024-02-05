using System.Linq;
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
using PureHDF.Selections;
using Serilog.Formatting.Display;
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
        
        public int[] ArrayStages { get; private set; }
        
        private ResourceVariable MeanInteractions { get; set; }
        
        public IAgentManager AgentManager { get; private set; }

        private LearnableParams _learnableParams;

        private Tensor _infectedIndex;
        
        private int _infinityTime;

        private ResourceVariable _infectedTime;

        private ResourceVariable _nextStageTimes;

        private const int ChildUpperIndex = 1;

        private const int AdultUpperIndex = 6;

        private readonly int[] _mu = { 2, 4, 3 };


        public override bool InitLayer(LayerInitData layerInitData, RegisterAgent registerAgentHandle,
            UnregisterAgent unregisterAgentHandle)
        {
            var initiated = base.InitLayer(layerInitData, registerAgentHandle, unregisterAgentHandle);
            _infinityTime = Steps + 1;
            ContactEnvironment = new ContactGraphEnvironment(AgentCount, Steps);
            AgeGroups = tf.Variable(tf.zeros(new Shape(AgentCount, 1), TF_DataType.TF_INT32));
            AgentManager = layerInitData.Container.Resolve<IAgentManager>();
            AgentManager.Spawn<Host, InfectionLayer>().ToList();
            _learnableParams = LearnableParams.Instance;
            InitStages();
            InitMeanInteractions();
            InitTimeVariables();
            // _infectedIndex = tf.expand_dims(tf.greater(tf.squeeze(Stages), tf.constant(0)), axis: 1);
            _infectedIndex = tf.greater(Stages, tf.constant(0));
            // tf.print(_infectedIndex);
            return initiated;
        }

        public void Tick()
        {
            var nodeFeatures =
                tf.concat(
                    new[] { AgeGroups, tf.stop_gradient(Stages), tf.cast(_infectedIndex, TF_DataType.TF_INT32), _infectedTime, MeanInteractions },
                    axis: 1);
            var exposedToday = ContactEnvironment.Forward(nodeFeatures, (int)Context.CurrentTick);
            var nextStages = updateStages(exposedToday);
            _nextStageTimes = tf.Variable(UpdateNextStageTimes(exposedToday));
            Stages = tf.Variable(nextStages);
            _infectedIndex = tf.where(tf.cast(exposedToday, TF_DataType.TF_BOOL), tf.fill(tf.shape(_infectedIndex), tf.constant(true)), _infectedIndex);
            _infectedTime = tf.Variable(tf.where(tf.cast(exposedToday, TF_DataType.TF_BOOL), tf.fill(tf.shape(_infectedTime), tf.constant((int)Context.CurrentTick)), _infectedTime));
            ArrayStages = Stages.numpy().ToArray<int>();
        }

        public void PreTick()
        { 
            
        }

        public void PostTick()
        {
            
        }

        private Tensor UpdateNextStageTimes(Tensor exposedToday)
        {
            var newTransitionTimes = tf.identity(_nextStageTimes);
            var currentStages = tf.identity(Stages);
            var conditionInfectedAndTransitionTime = tf.logical_and(tf.equal(currentStages, tf.constant((int)Stage.Infected)), 
                tf.equal(_nextStageTimes, tf.constant((int)Context.CurrentTick)));
            
            newTransitionTimes = tf.where(conditionInfectedAndTransitionTime, 
                tf.fill(tf.shape(newTransitionTimes), tf.constant(_infinityTime)), newTransitionTimes);
            
            var conditionExposedAndTransitionTime = tf.logical_and(tf.equal(currentStages, tf.constant((int)Stage.Exposed)), 
                tf.equal(_nextStageTimes, tf.constant((int)Context.CurrentTick)));
            
            newTransitionTimes = tf.where(conditionExposedAndTransitionTime,
                (tf.fill(tf.shape(newTransitionTimes),tf.constant((int)Context.CurrentTick) + 
                                                      _learnableParams.InfectedToRecoveredTime)), newTransitionTimes);
            // tf.print(tf.shape(_nextStageTimes));
            var result = exposedToday * (tf.constant((int)Context.CurrentTick + 1) + _learnableParams.ExposedToInfectedTime)
                + (tf.fill(tf.shape(exposedToday), tf.constant(1)) - exposedToday) * newTransitionTimes;
            return result;
        }

        private Tensor updateStages(Tensor exposedToday)
        {
            var transitionToInfected = tf.Variable(tf.cast(tf.less_equal(_nextStageTimes, (int)Context.CurrentTick), 
                TF_DataType.TF_INT32) * (int)Stage.Infected) + tf.Variable(tf.cast(tf.greater(_nextStageTimes, 
                (int)Context.CurrentTick), TF_DataType.TF_INT32) * (int)Stage.Exposed);
            // softmax mit p als learnable param mortality
            var transitionToMortalityOrRecovered = tf.Variable(tf.cast(tf.less_equal(_nextStageTimes, (int)Context.CurrentTick), 
                TF_DataType.TF_INT32) * (int)Stage.Recovered) + tf.Variable(tf.cast(tf.greater(_nextStageTimes, 
                (int)Context.CurrentTick), TF_DataType.TF_INT32) * (int)Stage.Infected);
            var probabilityMortality = tf.cast(tf.logical_and(tf.equal(Stages, tf.constant((int)Stage.Infected)), 
                tf.less_equal(_nextStageTimes, (int)Context.CurrentTick)), dtype: TF_DataType.TF_INT32) * _learnableParams.MortalityRate;
            // tf.print(probabilityMortality);
            var p = tf.concat(new [] { probabilityMortality, 1 - probabilityMortality }, axis: 1);
            var mortality = tf.cast(GumbelSoftmax.Execute(tf.Variable(p))[Slice.All, 0], dtype: TF_DataType.TF_BOOL);
            // tf.print(transitionToMortalityOrRecovered);
            transitionToMortalityOrRecovered = tf.where(mortality, tf.fill(tf.shape(transitionToMortalityOrRecovered), 
                (int)Stage.Mortality), transitionToMortalityOrRecovered);
            // tf.print(transitionToMortalityOrRecovered);
            var stageProgression =
                tf.cast(tf.equal(Stages, tf.Variable((int)Stage.Susceptible)), TF_DataType.TF_INT32) *
                (int)Stage.Susceptible +
                tf.cast(tf.equal(Stages, tf.Variable((int)Stage.Recovered)), TF_DataType.TF_INT32) *
                (int)Stage.Recovered +
                tf.cast(tf.equal(Stages, tf.Variable((int)Stage.Mortality)), TF_DataType.TF_INT32) *
                (int)Stage.Mortality +
                tf.cast(tf.equal(Stages, tf.Variable((int)Stage.Exposed)), TF_DataType.TF_INT32) *
                transitionToInfected +
                tf.cast(tf.equal(Stages, tf.Variable((int)Stage.Infected)), TF_DataType.TF_INT32) *
                transitionToMortalityOrRecovered;

            var nextStages = exposedToday * (int)Stage.Exposed + stageProgression;
            return nextStages;
        }
        
        private void InitStages()
        {
            var probabilityInfected = _learnableParams.InitialInfectionRate / tf.constant(100) * tf.ones(new Shape(AgentCount, 1));
            var p = tf.Variable(tf.concat(new [] { probabilityInfected, 1 - probabilityInfected }, axis: 1));
            Stages = tf.Variable(tf.expand_dims(tf.cast(GumbelSoftmax.Execute(p)[Slice.All,0], dtype: TF_DataType.TF_INT32), axis: 1) * 2);
        }

        private void InitMeanInteractions()
        {
            MeanInteractions = tf.Variable(tf.zeros(new Shape(AgentCount, 1)), dtype: TF_DataType.TF_INT32);
            var childAgents = tf.less_equal(AgeGroups, ChildUpperIndex);
            var adultAgents = tf.logical_and(tf.greater(AgeGroups, ChildUpperIndex), tf.less_equal(AgeGroups, AdultUpperIndex));
            var elderlyAgents = tf.greater(AgeGroups, AdultUpperIndex);
            MeanInteractions = tf.Variable(MeanInteractions.assign(tf.where(
                childAgents, tf.fill(tf.shape(MeanInteractions), _mu[0]),
                MeanInteractions)));
            MeanInteractions = tf.Variable(MeanInteractions.assign(tf.where(
                adultAgents, tf.fill(tf.shape(MeanInteractions), _mu[1]),
                MeanInteractions)));
            MeanInteractions = tf.Variable(MeanInteractions.assign(tf.where(
                elderlyAgents, tf.fill(tf.shape(MeanInteractions), _mu[2]),
                MeanInteractions)));
        }
        
        private void InitTimeVariables()
        {
            _infectedTime = tf.Variable(_infinityTime * tf.ones(new Shape(AgentCount, 1), dtype: TF_DataType.TF_INT32));
            _nextStageTimes = tf.Variable(_infinityTime * tf.ones(new Shape(AgentCount, 1), dtype: TF_DataType.TF_INT32));
            var exposedCondition = tf.equal(Stages, tf.Variable((int)Stage.Exposed, dtype: TF_DataType.TF_INT32));
            // var infectedCondition = tf.equal(tf.squeeze(Stages), tf.Variable((int)Stage.Infected, dtype: TF_DataType.TF_INT32));
            var infectedCondition = tf.equal(Stages, tf.Variable((int)Stage.Infected, dtype: TF_DataType.TF_INT32));
            // tf.print(infectedCondition);
            _infectedTime = tf.Variable(tf.where(exposedCondition, tf.fill(tf.shape(_infectedTime), tf.Variable(0, dtype:TF_DataType.TF_INT32)), _infectedTime));
            _infectedTime = tf.Variable(tf.where(infectedCondition, tf.fill(tf.shape(_infectedTime), (tf.Variable(-1, dtype:TF_DataType.TF_INT32) * _learnableParams.ExposedToInfectedTime) + tf.Variable(1, dtype:TF_DataType.TF_INT32)), _infectedTime));
            
            _nextStageTimes = tf.Variable(tf.where(exposedCondition, tf.fill(tf.shape(_nextStageTimes), (_learnableParams.ExposedToInfectedTime) + tf.Variable(1, dtype:TF_DataType.TF_INT32)), _nextStageTimes));
            _nextStageTimes = tf.Variable(tf.where(infectedCondition, tf.fill(tf.shape(_nextStageTimes), (_learnableParams.InfectedToRecoveredTime) + tf.Variable(1, dtype:TF_DataType.TF_INT32)), _nextStageTimes));
        }
    }
}