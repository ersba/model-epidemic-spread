using System;
using System.IO;
using EpidemicSpread.Model;
using Mars.Components.Starter;
using Mars.Interfaces.Model;

namespace EpidemicSpread
{
    internal static class Program
    {
        private static void Main()
        {
            var description = new ModelDescription();
            
            description.AddLayer<InfectionLayer>();
            
            description.AddAgent<Host, InfectionLayer>();
            
            // use config.json 
            var file = File.ReadAllText("config.json");
            var config = SimulationConfig.Deserialize(file);

            var starter = SimulationStarter.Start(description, config);
            var handle = starter.Run();
            Console.WriteLine("Successfully executed iterations: " + handle.Iterations);
            
            starter.Dispose();
        }
    }
}