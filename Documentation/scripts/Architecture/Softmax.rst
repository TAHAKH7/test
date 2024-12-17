

Softmax in Transformers
=======================

.. figure:: /Documentation/images/softmax2.jpg
    :width: 400
    :align: center
    :alt: Alternative Text






.. raw:: html
      
   <p style="text-align: justify;"><span style="color:#000080;"><i>

    In transformers, the softmax function is commonly used as part of the mechanism for calculating attention scores, which are critical for the self-attention mechanism that forms the basis of the model. It is essential for several reasons:
   </i></span></p>

.. figure:: /Documentation/images/softmax1.jpg
    :width: 400
    :align: center
    :alt: Alternative Text


.. raw:: html
      
   <p style="text-align: justify;"><span style="color:#000080;">

    <p style="text-align: justify;">
    
     -<span style="color:blue;"> Attention Weights</span>: <span style="color:#000080;"><i> Transformers use attention mechanisms to weigh the importance of different input tokens when generating an output. Softmax is used to convert the raw attention scores, often called “logits,” into a probability distribution over the input tokens. This distribution assigns higher attention weights to more relevant tokens and lower weights to less relevant ones.
    
    </i></span></p>
    <p style="text-align: justify;">
     - <span style="color:blue;"> Probability Distribution</span>:<span style="color:#000080;"><i> Softmax ensures that the attention scores are transformed into a valid probability distribution, with all values between 0 and 1 and the sum equal to 1. This property is important for correctly weighing the input tokens while taking into account their relative importance.
    
    </span></p>
    <p style="text-align: justify;">
     - <span style="color:blue;"> Stabilizing Gradients</span>:<span style="color:#000080;"><i>The softmax function has a smooth gradient, which makes it easier to train deep neural networks like transformers using techniques like backpropagation. It helps with gradient stability during training, making it easier for the model to learn and adjust its parameters.
    </i></span></p>
    
    <p style="text-align: justify;"><span style="color:#000080;"><i>

    The softmax function is typically applied to the raw attention scores obtained from the dot product of query and key vectors in the self-attention mechanism. The formula for computing the softmax attention weights for a given query token in a transformer is as follows:
   </i></span></p>

.. figure:: /Documentation/images/softmax.jpg
    :width: 500
    :align: center
    :alt: Alternative Text


.. math::

   \text{Softmax}(QK^\top) = \frac{\exp(QK_i^\top)}{\sum_j \exp(QK_j^\top)}


.. raw:: html
      
   <p style="text-align: justify;"><span style="color:#000080;">

    <i>Here</i>, <span style="color:red;"><strong> Q </strong></span> <i>represents the query vector,</i> <span style="color:red;"><strong>K</strong></span><i> represents the key vectors of the input tokens, and the exponential function (\exp) is used to transform the raw scores into positive values. The denominator ensures that the resulting values form a probability distribution.</i>
   </span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>

    In summary, the softmax function is a crucial component of transformers that enables them to learn how to weigh input tokens based on their relevance to the current context, making the model’s self-attention mechanism effective in capturing dependencies and relationships in the data.
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>

    And the most important thing is the softmax is used to prevent exploding gradient or vanishing gradient problems.
   </i></span></p>


.. admonition::  For more information

   .. container:: blue-box
    
    * `"why do we use softmax in transformers" <https://medium.com/@maitydi567/why-do-we-use-softmax-in-transformers-fdfd50f5f4c1#:~:text=In%20summary%2C%20the%20softmax%20function,and%20relationships%20in%20the%20data.>`__
    
    * `"softmax paper link" <https://arxiv.org/pdf/2207.03341.pdf>`__







