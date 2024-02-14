using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpread
{
    public static class Params
    {
        public static readonly float ChildUpperIndex = 1f;
        
        public static readonly float AdultUpperIndex = 6f;
        
        public static readonly float[] Mu = { 2f, 4f, 3f };

        public static int Steps = 0;

        public static int AgentCount = 0;
        
        public static Tensor EdgeAttribute = tf.constant(1f);
        
        public static Tensor Susceptibility = tf.constant(new float[] {0.35f, 0.69f, 1.03f, 1.03f, 1.03f, 1.03f, 1.27f, 1.52f});
        
        public static Tensor Infector = tf.constant(new float[] {0.0f, 0.33f, 0.72f, 0.0f, 0.0f});
    }
}