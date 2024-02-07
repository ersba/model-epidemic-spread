using System;
using System.IO;
using System.Linq;
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
            Console.WriteLine(config);
            Params.Steps = (int)(config.Globals.Steps?? 0);
            Params.AgentCount = config.AgentMappings[0].InstanceCount ?? 0;
            var starter = SimulationStarter.Start(description, config);
            var handle = starter.Run();
            Console.WriteLine("Successfully executed iterations: " + handle.Iterations);
            
            starter.Dispose();
        }
    }
}