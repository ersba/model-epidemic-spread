using System;
using System.Linq;
using System.IO;
using EpidemicSpread.Model;
using Mars.Components.Starter;
using Mars.Interfaces.Model;
using Tensorflow;

namespace EpidemicSpread
{
    internal static class Program
    
    {
        private static void Main()
        {
            var calibNn = new SimpleCalibNn();
            calibNn.Train(100);

            // EpidemicSpreadSimulation();
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
            
            var starter = SimulationStarter.Start(description, config);
            var handle = starter.Run();
            var deaths = ((InfectionLayer)handle.Model.AllActiveLayers.First()).Deaths;
            starter.Dispose();
            Console.WriteLine("Successfully executed iterations: " + handle.Iterations);
            return deaths;
        }
    }
}