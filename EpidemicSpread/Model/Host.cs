using System;
using System.Linq;
using Mars.Common;
using Mars.Common.Core.Random;
using Mars.Interfaces.Agents;
using Mars.Interfaces.Annotations;
using NumSharp;
using Mars.Interfaces.Environments;
using Mars.Interfaces.Layers;
using Mars.Numerics;
using Tensorflow;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.NumPy;
using Shape = Tensorflow.Shape;

namespace EpidemicSpread.Model 
{
    public class Host : IAgent<InfectionLayer>
    {
        [PropertyDescription]
        public int Line { get; set; }
        
        [PropertyDescription]
        public int AgeGroup { get; set; }
        
        [PropertyDescription]
        public int Stage { get; set; }
        
        [PropertyDescription] 
        public UnregisterAgent UnregisterHandle { get; set; }

        public void Init(InfectionLayer layer)
        {
            _infectionlayer = layer;
            // Console.WriteLine(AgeGroup);
            _infectionlayer.AgeGroups[Line].assign(tf.constant(AgeGroup)); 
        }

        public void Tick()
        {
            
        }

        private void Interact()
        {
            
        }

        private void Die()
        {
            UnregisterHandle.Invoke(_infectionlayer, this);
        }

        private InfectionLayer _infectionlayer;

        public Guid ID { get; set; }
    }
    
    
}