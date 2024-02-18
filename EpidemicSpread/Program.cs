using System;
using System.Linq;
using System.IO;
using System.Linq;
using EpidemicSpread.Model;
using System.Globalization;
using Mars.Common.Core;
using Mars.Components.Layers;
using Mars.Components.Starter;
using Mars.Core.Simulation.Entities;
using Mars.Interfaces.Model;
using Tensorflow;
using Xla;
using static Tensorflow.Binding;

namespace EpidemicSpread
{
    internal static class Program
    
    {
        private static Tensor Deaths { get; set; }
        private static void Main()
        { 
            var calibNn = new SimpleCalibNn();
            calibNn.Train(800);
            
            // var t = tf.constant(1.0f, dtype: TF_DataType.TF_FLOAT);
            // tf.print(t);
            // tf.print(tf.cast(t, TF_DataType.TF_BOOL));
            
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
            Console.WriteLine(deaths);
            starter.Dispose();
            Console.WriteLine("Successfully executed iterations: " + handle.Iterations);
            return deaths;
        }
    }
}