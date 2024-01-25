using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpread
{
    public static class GumbelSoftmax
    {
        public static Tensor Execute(ResourceVariable probabilities, double temperature = 1.0)
        {
            var gumbelNoise = -tf.math.log(-tf.math.log(tf.random.uniform(probabilities.shape)));
            var softSample = tf.nn.softmax((tf.math.log(probabilities + 1e-9) + gumbelNoise) / temperature);
            var hardSample = tf.cast(tf.equal(softSample, tf.reduce_max(softSample, axis: 1, keepdims: true)),softSample.dtype);
            return tf.stop_gradient(hardSample - softSample) + softSample;
        }
    }
}

