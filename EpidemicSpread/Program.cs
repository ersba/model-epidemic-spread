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
        public static Tensor Deaths;
        
        private static SimulationStarter _starter;
        
        private static SimulationWorkflowState _handle;
        private static void Main()
        {
            for (int i = 0; i < 1; i++)
            {
                var calibNn = new SimpleCalibNn();
                calibNn.Train(100);
            }
            
    
            // var error = tf.square(newDeaths - tf.constant(100f));
            //     var grad = g.gradient(error, Deaths);
            //     Console.WriteLine(grad);


                // Console.WriteLine("Successfully executed iterations: " + _handle.Iterations);
            // }

            // _starter.Dispose();
        }

        private static Tensor Computation(ResourceVariable deaths)
        {
            return deaths * 2f;
        }
        public static Tensor EpidemicSpreadSimulation()  
        {
            var description = new ModelDescription();
            description.AddLayer<TestLayer>();
            
            var file = File.ReadAllText("config.json");
            var config = SimulationConfig.Deserialize(file);
            Params.Steps = (int)(config.Globals.Steps ?? 0);
            Params.AgentCount = config.AgentMappings[0].InstanceCount ?? 0;
            
            var starter = SimulationStarter.Start(description, config);
            var handle = starter.Run();
            Deaths = ((TestLayer)handle.Model.AllActiveLayers.First()).Deaths;
            starter.Dispose();
            Console.WriteLine("Successfully executed iterations: " + handle.Iterations);
            return Deaths;
        }

        public static void Dispose()
        {
            _starter.Dispose();
        }
    }
}