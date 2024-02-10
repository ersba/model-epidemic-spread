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
        public static ResourceVariable Deaths;
        
        private static SimulationStarter _starter;
        
        private static SimulationWorkflowState _handle;
        private static void Main()
        {
            
            // var calibNn = new SimpleCalibNn();
            using (var g = tf.GradientTape())
            {
                Deaths = tf.Variable(10f);
                g.watch(Deaths);
            // var description = new ModelDescription();
            // description.AddLayer<InfectionLayer>();
            // description.AddAgent<Host, InfectionLayer>();
            //
            // var file = File.ReadAllText("config.json");
            // var config = SimulationConfig.Deserialize(file);
            // Params.Steps = (int)(config.Globals.Steps ?? 0);
            // Params.AgentCount = config.AgentMappings[0].InstanceCount ?? 0;
            //
            // _starter = SimulationStarter.Start(description, config);
            // _handle = _starter.Run();
            
            // Deaths = ((InfectionLayer)_handle.Model.AllActiveLayers.First()).Deaths;

            var newDeaths = Computation(Deaths);
    
            var error = tf.square(newDeaths - tf.constant(100f));
                var grad = g.gradient(error, Deaths);
                Console.WriteLine(grad);
                // calibNn.Train(1);

                Console.WriteLine("Successfully executed iterations: " + _handle.Iterations);
            }

            // _starter.Dispose();
        }

        private static Tensor Computation(ResourceVariable deaths)
        {
            return deaths * 2f;
        }
        public static Tensor EpidemicSpreadSimulation()  
        {
            var description = new ModelDescription();
            description.AddLayer<InfectionLayer>();
            description.AddAgent<Host, InfectionLayer>();
            
            var file = File.ReadAllText("config.json");
            var config = SimulationConfig.Deserialize(file);
            Params.Steps = (int)(config.Globals.Steps ?? 0);
            Params.AgentCount = config.AgentMappings[0].InstanceCount ?? 0;
            
            _starter = SimulationStarter.Start(description, config);
            _handle = _starter.Run();
            var targetValue = ((InfectionLayer)_handle.Model.AllActiveLayers.First()).Deaths;
            Console.WriteLine("Successfully executed iterations: " + _handle.Iterations);
            return targetValue;
        }

        public static void Dispose()
        {
            _starter.Dispose();
        }
    }
}