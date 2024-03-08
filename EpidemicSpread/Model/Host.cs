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
        public int Index { get; set; }
        
        [PropertyDescription]
        public int MyAgeGroup { get; set; }
        
        [PropertyDescription]
        public int MyStage { get; set; }
        
        [PropertyDescription] 
        public UnregisterAgent UnregisterHandle { get; set; }

        public void Init(InfectionLayer layer)
        {
            _infectionLayer = layer;
            _infectionLayer.ArrayAgeGroups[Index] = MyAgeGroup; 
        }

        public void Tick()
        {
            Interact();
            if (MyStage != (int)Stage.Recovered) Progress();
            if (MyStage == (int)Stage.Mortality) Die();
        }

        private void Interact()
        {
            if (_infectionLayer.ContactEnvironment.Interact(Index)) MyStage = (int)Stage.Exposed;
        }

        private void Die()
        {
           // UnregisterHandle.Invoke(_infectionLayer, this);
        }

        private void Progress()
        {
            MyStage = _infectionLayer.ArrayStages[Index];
            // if (MyStage == (int)Stage.Recovered) Console.WriteLine("I'm recovered!!!");
        }

        private InfectionLayer _infectionLayer;

        public Guid ID { get; set; }
    }
    
    
}