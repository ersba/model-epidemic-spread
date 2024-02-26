using System;
using System.Linq;
using System.IO;
using System.Linq;
using EpidemicSpread.Model;
using System.Globalization;
using Mars.Common.Core;
using Mars.Components.Layers;
using Mars.Components.Starter;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Mars.Core.Simulation.Entities;
using Mars.Interfaces.Model;
using Tensorflow;
using Xla;
using static Tensorflow.Binding;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras;

namespace EpidemicSpread
{
    internal static class Program
    
    {
        private static void Main()
        { 
            var calibNn = new SimpleCalibNn();
            calibNn.CustomTrain(100);

            // EpidemicSpreadSimulation();
        }
        public static Tensor EpidemicSpreadSimulation()  
        {
            var description = new ModelDescription();
            // description.AddLayer<TestLayer>();
            description.AddLayer<InfectionLayer>();
            description.AddAgent<Host, InfectionLayer>();
            
            var file = File.ReadAllText("config.json");
            var config = SimulationConfig.Deserialize(file);
            Params.Steps = (int)(config.Globals.Steps ?? 0);
            Params.AgentCount = config.AgentMappings[0].InstanceCount ?? 0;
            
            var starter = SimulationStarter.Start(description, config);
            var handle = starter.Run();
            // var deaths = ((TestLayer)handle.Model.AllActiveLayers.First()).Deaths;
            var deaths = ((InfectionLayer)handle.Model.AllActiveLayers.First()).Deaths;
            starter.Dispose();
            Console.WriteLine("Successfully executed iterations: " + handle.Iterations);
            return deaths;
        }
    }
}