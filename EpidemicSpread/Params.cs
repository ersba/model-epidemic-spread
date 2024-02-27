using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpread
{
    public static class Params
    {
        public static readonly int ChildUpperIndex = 1;
        
        public static readonly int AdultUpperIndex = 6;
        
        public static readonly int[] Mu = { 2, 4, 3 };

        public static int Steps = 0;

        public static int AgentCount = 0;
        
        public static Tensor EdgeAttribute = tf.constant(1f);
        
        public static Tensor Susceptibility = tf.constant(new float[] {0.35f, 0.69f, 1.03f, 1.03f, 1.03f, 1.03f, 1.27f, 1.52f});
        
        public static Tensor Infector = tf.constant(new float[] {0.0f, 0.33f, 0.72f, 0.0f, 0.0f});
    }
}