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
        public ResourceVariable InitialInfectionRate { get; set; }
        public ResourceVariable R0Value { get; set; }
        public ResourceVariable MortalityRate { get; set; }
        public ResourceVariable ExposedToInfectedTime { get; set; }
        public ResourceVariable InfectedToRecoveredTime { get; set; }
        
        private LearnableParams()
        {
            InitialInfectionRate = tf.Variable(50, dtype: TF_DataType.TF_FLOAT);
            R0Value = tf.Variable(5.18, dtype: TF_DataType.TF_FLOAT);
            MortalityRate = tf.Variable(0.01, dtype: TF_DataType.TF_FLOAT);
            ExposedToInfectedTime = tf.Variable(3, dtype: TF_DataType.TF_INT32);
            InfectedToRecoveredTime = tf.Variable(5, dtype: TF_DataType.TF_INT32);
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