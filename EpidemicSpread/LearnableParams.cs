using Tensorflow;
using Tensorflow.Keras.Utils;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.NumPy;

namespace EpidemicSpread
{
    public class LearnableParams
    {
        private static LearnableParams _instance;
        public Tensor InitialInfectionRate { get; set; }
        public Tensor R0Value { get; set; }
        public Tensor MortalityRate { get; set; }
        public Tensor ExposedToInfectedTime { get; set; }
        public Tensor InfectedToRecoveredTime { get; set; }
        
        public Tensor TestTensor { get; set; }
        
        private LearnableParams()
        {
            InitialInfectionRate = tf.constant(0.05, dtype: TF_DataType.TF_FLOAT);
            R0Value = tf.constant(5.18, dtype: TF_DataType.TF_FLOAT);
            MortalityRate = tf.constant(0.1, dtype: TF_DataType.TF_FLOAT);
            ExposedToInfectedTime = tf.constant(3, dtype: TF_DataType.TF_INT32);
            InfectedToRecoveredTime = tf.constant(5, dtype: TF_DataType.TF_INT32);
        }
        
        public static LearnableParams Instance
        {
            get
            {
                if (_instance == null)
                {
                    _instance = new LearnableParams();
                }
                return _instance;
            }
        }
    }
}