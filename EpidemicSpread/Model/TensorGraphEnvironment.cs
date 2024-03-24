using System;
using Tensorflow;
using static Tensorflow.Binding;

namespace EpidemicSpread.Model;

public abstract class TensorGraphEnvironment
{
    private Tensor Aggregate(Tensor messages, Tensor targetIndices, int numNodes)
    {
        // [4180] Messages
        // [1000] Nodes
        // [4180 1000]
        var oneHotIndices = tf.one_hot(targetIndices, depth: numNodes);
        var expandedMessages = tf.expand_dims(messages, axis: -1);
        var weightedMessages = expandedMessages * oneHotIndices;
        
        var aggregatedMessages = tf.reduce_sum(weightedMessages, axis: 0);
        tf.print(tf.shape(aggregatedMessages));
        return aggregatedMessages;
    }

    protected Tensor Propagate(Tensor edgeIndex, Tensor nodeFeatures, params object[] args)
    {
        // edgeIndex: [2, 4180]
        // collect all infected indices for each target Node in a tensor
        var sourceNode = tf.gather(edgeIndex, tf.constant(0));
        var targetNode = tf.gather(edgeIndex, tf.constant(1));
        var sourceFeature = tf.gather(nodeFeatures, sourceNode);
        var targetFeature = tf.gather(nodeFeatures, targetNode);
        
        // calculation of the transmission chance(message) for every edge
        var edgeMessages = Message(sourceFeature, targetFeature, (int) args[0]);
        // allocation of the messages to the target nodes
        var outFeatures = Aggregate(edgeMessages, targetNode, (int)nodeFeatures.shape[0]);
        return outFeatures;
        
        
    }

    protected abstract Tensor Message(Tensor sourceNode, Tensor targetNode, int currentTick);
}