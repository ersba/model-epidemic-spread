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
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.NumPy;

namespace EpidemicSpread.Model

{
    public class InfectionLayer : AbstractLayer, ISteppedActiveLayer
    {
        public ContactGraphEnvironment ContactEnvironment { get; private set; }
        
        public Tensor AgeGroups { get; private set; }
        
        public Tensor Stages { get; private set; }
        
        public Tensor Deaths { get; private set; }
        
        public int[] ArrayStages { get; private set; }
        
        public int[] ArrayAgeGroups { get; private set; }
        
        private Tensor MeanInteractions { get; set; }
        
        public IAgentManager AgentManager { get; private set; }

        private LearnableParams _learnableParams;

        private Tensor _infectedIndex;
        
        private int _infinityTime;

        private Tensor _infectedTime;

        private Tensor _nextStageTimes;
        


        public override bool InitLayer(LayerInitData layerInitData, RegisterAgent registerAgentHandle,
            UnregisterAgent unregisterAgentHandle)
        {
            var initiated = base.InitLayer(layerInitData, registerAgentHandle, unregisterAgentHandle);
            _infinityTime = Params.Steps + 1;
            ContactEnvironment = new ContactGraphEnvironment();
            ArrayAgeGroups = new int[Params.AgentCount];
            AgentManager = layerInitData.Container.Resolve<IAgentManager>();
            AgentManager.Spawn<Host, InfectionLayer>().ToList();
            AgeGroups = tf.expand_dims(tf.constant(ArrayAgeGroups, dtype: TF_DataType.TF_FLOAT), axis: 1);
            _learnableParams = LearnableParams.Instance;
            InitStages();
            ArrayStages = tf.cast(Stages, TF_DataType.TF_INT32).numpy().ToArray<int>();
            InitMeanInteractions();
            InitTimeVariables();
            // _infectedIndex = tf.expand_dims(tf.greater(tf.squeeze(Stages), tf.constant(0)), axis: 1);
            _infectedIndex = tf.greater(Stages, tf.constant(0, TF_DataType.TF_FLOAT));
            Deaths = tf.constant(0f, TF_DataType.TF_FLOAT);
            // tf.print(_infectedIndex);
            
            return initiated;
        }

        public void Tick()
        {
            // Computation of Gradient
            var recoveredAndDead= Stages * tf.logical_and(tf.equal(Stages, tf.constant(Stage.Infected, TF_DataType.TF_FLOAT)),
                tf.less_equal(_nextStageTimes, (float) Context.CurrentTick)) / (float) Stage.Infected;
            
            Deaths += tf.reduce_sum(recoveredAndDead) * (_learnableParams.MortalityRate);
            // var nodeFeatures =
            //     tf.concat(
            //         new[] { AgeGroups, tf.stop_gradient(Stages), tf.cast(_infectedIndex, TF_DataType.TF_FLOAT), _infectedTime, MeanInteractions },
            //         axis: 1);
            // // var exposedToday = ContactEnvironment.Forward(nodeFeatures, (int)Context.CurrentTick);
            // var exposedToday = tf.ones_like(Stages, TF_DataType.TF_FLOAT);
            var exposedToday = tf.zeros_like(Stages);
            var nextStages = UpdateStages(exposedToday);
            _nextStageTimes = UpdateNextStageTimes(exposedToday);
            Stages = nextStages;
            // _infectedIndex = tf.where(tf.cast(exposedToday, TF_DataType.TF_BOOL), tf.fill(tf.shape(_infectedIndex), tf.constant(true)), _infectedIndex);
            // _infectedTime = tf.where(tf.cast(exposedToday, TF_DataType.TF_BOOL), tf.fill(tf.shape(_infectedTime), tf.constant((float)Context.CurrentTick)), _infectedTime);
            ArrayStages = tf.cast(Stages, TF_DataType.TF_INT32).numpy().ToArray<int>();
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
            var conditionInfectedAndTransitionTime = tf.logical_and(tf.equal(currentStages, tf.constant((float)Stage.Infected)), 
                tf.equal(_nextStageTimes, tf.constant((float)Context.CurrentTick)));
            
            newTransitionTimes = tf.where(conditionInfectedAndTransitionTime, 
                tf.fill(tf.shape(newTransitionTimes), tf.constant(_infinityTime, TF_DataType.TF_FLOAT)), newTransitionTimes);
            
            var conditionExposedAndTransitionTime = tf.logical_and(tf.equal(currentStages, tf.constant((float)Stage.Exposed)), 
                tf.equal(_nextStageTimes, tf.constant((float)Context.CurrentTick)));
            
            newTransitionTimes = tf.where(conditionExposedAndTransitionTime,
                (tf.fill(tf.shape(newTransitionTimes),tf.constant((float)Context.CurrentTick) + 
                                                      _learnableParams.InfectedToRecoveredTime)), newTransitionTimes);
            // tf.print(tf.shape(_nextStageTimes));
            var result = exposedToday * (tf.constant((float)Context.CurrentTick + 1) + _learnableParams.ExposedToInfectedTime)
                + (tf.fill(tf.shape(exposedToday), tf.constant(1)) - exposedToday) * newTransitionTimes;
            return result;
        }

        private Tensor UpdateStages(Tensor exposedToday)
        {
            var transitionToInfected = tf.cast(tf.less_equal(_nextStageTimes, (float)Context.CurrentTick), 
                TF_DataType.TF_FLOAT) * (float)Stage.Infected + tf.cast(tf.greater(_nextStageTimes, 
                (float)Context.CurrentTick), TF_DataType.TF_FLOAT) * (float)Stage.Exposed;
            var transitionToMortalityOrRecovered = tf.cast(tf.less_equal(_nextStageTimes, (float)Context.CurrentTick), 
                TF_DataType.TF_FLOAT) * (float)Stage.Recovered + tf.cast(tf.greater(_nextStageTimes, 
                (float)Context.CurrentTick), TF_DataType.TF_FLOAT) * (float)Stage.Infected;
            var probabilityMortality = tf.cast(tf.logical_and(tf.equal(Stages, tf.constant((float)Stage.Infected)), 
                tf.less_equal(_nextStageTimes, (float)Context.CurrentTick)), dtype: TF_DataType.TF_FLOAT) * (_learnableParams.MortalityRate/100);
            // tf.print(probabilityMortality);
            var p = tf.concat(new [] { probabilityMortality, 1 - probabilityMortality }, axis: 1);
            var mortality = tf.cast(GumbelSoftmax.Execute(p)[Slice.All, 0], dtype: TF_DataType.TF_BOOL);
            // tf.print(transitionToMortalityOrRecovered);
            transitionToMortalityOrRecovered = tf.where(mortality, tf.fill(tf.shape(transitionToMortalityOrRecovered), 
                (float)Stage.Mortality), transitionToMortalityOrRecovered);
            // tf.print(transitionToMortalityOrRecovered);
            var stageProgression =
                tf.cast(tf.equal(Stages, tf.constant((float)Stage.Susceptible)), TF_DataType.TF_FLOAT) *
                (float)Stage.Susceptible +
                tf.cast(tf.equal(Stages, tf.constant((float)Stage.Recovered)), TF_DataType.TF_FLOAT) *
                (float)Stage.Recovered +
                tf.cast(tf.equal(Stages, tf.constant((float)Stage.Mortality)), TF_DataType.TF_FLOAT) *
                (float)Stage.Mortality +
                tf.cast(tf.equal(Stages, tf.constant((float)Stage.Exposed)), TF_DataType.TF_FLOAT) *
                transitionToInfected +
                tf.cast(tf.equal(Stages, tf.constant((float)Stage.Infected)), TF_DataType.TF_FLOAT) *
                transitionToMortalityOrRecovered;

            var nextStages = exposedToday * (float)Stage.Exposed + stageProgression;
            return nextStages;
        }
        
        private void InitStages()
        {
            var probabilityInfected = _learnableParams.InitialInfectionRate * tf.ones(new Shape(Params.AgentCount, 1));
            var p = tf.concat(new [] { probabilityInfected, 1 - probabilityInfected }, axis: 1);
            Stages = tf.expand_dims(tf.cast(GumbelSoftmax.Execute(p)[Slice.All,0], dtype: TF_DataType.TF_FLOAT), axis: 1) * 2;
        }

        private void InitMeanInteractions()
        {
            MeanInteractions = tf.zeros(new Shape(Params.AgentCount, 1), dtype: TF_DataType.TF_FLOAT);
            var childAgents = tf.less_equal(AgeGroups, Params.ChildUpperIndex);
            var adultAgents = tf.logical_and(tf.greater(AgeGroups, Params.ChildUpperIndex), tf.less_equal(AgeGroups, Params.AdultUpperIndex));
            var elderlyAgents = tf.greater(AgeGroups, Params.AdultUpperIndex);
            MeanInteractions = tf.where(childAgents, tf.fill(tf.shape(MeanInteractions), Params.Mu[0]),
                MeanInteractions);
            MeanInteractions = tf.where(adultAgents, tf.fill(tf.shape(MeanInteractions), Params.Mu[1]),
                MeanInteractions);
            MeanInteractions = tf.where(elderlyAgents, tf.fill(tf.shape(MeanInteractions), Params.Mu[2]),
                MeanInteractions);
        }
        
        private void InitTimeVariables()
        {
            _infectedTime = _infinityTime * tf.ones(new Shape(Params.AgentCount, 1), dtype: TF_DataType.TF_FLOAT);
            _nextStageTimes = _infinityTime * tf.ones(new Shape(Params.AgentCount, 1), dtype: TF_DataType.TF_FLOAT);
            var exposedCondition = tf.equal(Stages, tf.constant((float)Stage.Exposed, dtype: TF_DataType.TF_FLOAT));
            var infectedCondition = tf.equal(Stages, tf.constant((float)Stage.Infected, dtype: TF_DataType.TF_FLOAT));
            _infectedTime = tf.where(exposedCondition, tf.fill(tf.shape(_infectedTime), tf.constant(0, dtype:TF_DataType.TF_FLOAT)), _infectedTime);
            _infectedTime = tf.where(infectedCondition, tf.fill(tf.shape(_infectedTime), (tf.constant(-1, dtype:TF_DataType.TF_FLOAT) * _learnableParams.ExposedToInfectedTime) + tf.constant(1, dtype:TF_DataType.TF_FLOAT)), _infectedTime);
            
            _nextStageTimes = tf.where(exposedCondition, tf.fill(tf.shape(_nextStageTimes), (_learnableParams.ExposedToInfectedTime) + tf.constant(1, dtype:TF_DataType.TF_FLOAT)), _nextStageTimes);
            _nextStageTimes = tf.where(infectedCondition, tf.fill(tf.shape(_nextStageTimes), (_learnableParams.InfectedToRecoveredTime) + tf.constant(1, dtype:TF_DataType.TF_FLOAT)), _nextStageTimes);
        }
    }
}