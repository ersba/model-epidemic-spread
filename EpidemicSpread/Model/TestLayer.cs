using System.Linq;
using Mars.Common.Core.Random;
using Mars.Components.Environments;
using Mars.Components.Layers;
using Mars.Core.Data;
using Mars.Interfaces.Data;
using Mars.Interfaces.Environments;
using Mars.Interfaces.Layers;
using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpread.Model
{
    public class TestLayer : AbstractLayer, ISteppedActiveLayer
    {
        public IAgentManager AgentManager { get; private set; }
        
        public Tensor Deaths { get; private set; }
        
        private LearnableParams _learnableParams;

        public override bool InitLayer(LayerInitData layerInitData, RegisterAgent registerAgentHandle,
            UnregisterAgent unregisterAgentHandle)
        {
            var initiated = base.InitLayer(layerInitData, registerAgentHandle, unregisterAgentHandle);
            _learnableParams = LearnableParams.Instance;
            Deaths = tf.constant(0);
            return initiated;
        }



        public void Tick()
        {
            Deaths += _learnableParams.InfectedToRecoveredTime * 10;
        }

        public void PreTick()
        {
            
        }

        public void PostTick()
        {
            
        }
    }
}