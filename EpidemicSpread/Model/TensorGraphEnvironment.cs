using System;
using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpread.Model;

public abstract class TensorGraphEnvironment
{
    private Tensor Aggregate(Tensor messages, Tensor targetIndices, int numNodes)
    {
        var oneHotIndices = tf.one_hot(targetIndices, depth: numNodes);
        
        var expandedMessages = tf.expand_dims(messages, axis: -1);
        var weightedMessages = expandedMessages * oneHotIndices;
        
        var aggregatedMessages = tf.reduce_sum(weightedMessages, axis: 0);
        return aggregatedMessages;
    }

    protected Tensor Propagate(Tensor edgeIndex, Tensor nodeFeatures, params object[] args)
    {
        Tensor outFeatures = tf.Variable(tf.zeros_like(nodeFeatures));
        var sourceNode = tf.gather(edgeIndex, tf.constant(0));
        var targetNode = tf.gather(edgeIndex, tf.constant(1));
        var sourceFeature = tf.gather(nodeFeatures, sourceNode);
        var targetFeature = tf.gather(nodeFeatures, targetNode);

        var edgeMessages = Message(sourceFeature, targetFeature);
        outFeatures = Aggregate(edgeMessages, targetNode, (int)nodeFeatures.shape[0]);
        // tf.print(tf.shape(nodeFeatures));
        // tf.print(tf.shape(outFeatures));
        // tf.print(outFeatures);
        return outFeatures;
        
        
    }

    protected abstract Tensor Message(Tensor sourceNode, Tensor targetNode);
}