The Decoder
============

.. note::

   The decoder block is similar to the encoder block, except it calculates the source-target attention.


.. figure:: /Documentation/images/decoder.webp
   :width: 700
   :align: center
   :alt: Alternative Text


.. admonition::  Overview

   .. container:: greenish-gray-box

      .. raw:: html

         <p style="text-align: justify;"><span style="color:#000080;">

         A transformer decoder is a neural network architecture used in natural language processing tasks such as machine translation and text generation. It combines with an encoder to process input text and generate output text. It has multiple layers of self-attention and feed-forward neural networks. It is trained using a combination of supervised and unsupervised learning techniques. It is known for its accuracy and natural-sounding output.
         </span></p>
      

1. Introduction
------------------
.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">

   The transformer decoder is a crucial component of the Transformer architecture, which has revolutionized the field of natural language processing (NLP) in recent years. It is known for its state-of-the-art performance on various tasks, including machine translation, language modeling, and summarization. The transformer decoder works in conjunction with the encoder, which processes the input sequence and generates a sequence of contextualized representations known as "hidden states." These hidden states capture the meaning and context of the input sequence and are passed on to the transformer decoder.
   </span></p>

   <p style="text-align: justify;"><span style="color:#000080;">

   The transformer <strong>decoder block</strong> comprises <strong>multiple layers</strong> of <strong>self-attention</strong> and <strong>feed-forward</strong> neural networks, which work together to process the input and generate the output. The transformer decoder then uses these hidden states and the previously generated output tokens to predict the next output token and generate the final output sequence. This encoder-decoder architecture is necessary for NLP tasks as it allows for more accurate and natural-sounding output.
   </span></p>

   <p style="text-align: justify;"><span style="color:#000080;">

   As mentioned before, an input to the decoder is an output shifted right, which becomes a sequence of embeddings with positional encoding. So, we can think of the decoder block as another encoder generating enriched embeddings useful for translation outputs.
   </span></p>



.. figure:: /Documentation/images/DECODER1.png
   :width: 600
   :height: 500
   :align: center
   :alt: Alternative Text

.. note::

   The decoder and encoder share several similarities, but they differ in their input. We will explain this difference in the next section.


2. difference between decoder and encoder
------------------------------------------

.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">

   The first sublayer receives the previous output of the decoder stack, augments it with positional information, and implements multi-head self-attention over it. While the encoder is designed to attend to all words in the input sequence regardless of their position in the sequence, the decoder is modified to attend only to the preceding words. 
   Hence, the prediction for a word at position <strong> i </strong>can only depend on the known outputs for the words that come before it in the sequence. 
   In the multi-head attention mechanism (which implements multiple, single attention functions in parallel), this is achieved by introducing a mask over the values produced by the scaled multiplication of matrices <strong> Q </strong> and <strong> K </strong>.
   This masking is implemented by suppressing the matrix values that would otherwise correspond to illegal connections:
   </span></p>


.. figure:: /Documentation/images/mask.jpg
   :width: 600
   :align: center
   :alt: Alternative Text



3. Masked multi-head attention
--------------------------------

.. figure:: /Documentation/images/mask_att.jpg
   :width: 700
   :align: center
   :alt: Alternative Text

   
.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">

   means the multi-head attention receives inputs with masks so that the attention mechanism does not use information from the hidden (masked) positions. The paper mentions that they used the mask inside the attention calculation by setting attention scores to negative infinity (or a very large negative number). The softmax within the attention mechanisms effectively assigns zero probability to masked positions.
   </span></p>

   <p style="text-align: justify;"><span style="color:#000080;">

   Intuitively, it is as if we were gradually increasing the visibility of input sentences by the masks:
   </span></p>

.. figure:: /Documentation/images/maskk1.jpg
   :width: 500
   :align: center
   :alt: Alternative Text

.. figure:: /Documentation/images/maskk.jpg
   :width: 500
   :align: center
   :alt: Alternative Text


.. note::
   .. raw:: html

      <span style="color:#000080;">Causal Model:</span> The model must not be able to see the future words


4. Multi-Head Attention
-------------------------
.. figure:: /Documentation/images/encoder_decoder.jpg
   :width: 700
   :align: center
   :alt: Alternative Text

.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
   
   The second layer implements a multi-head self-attention mechanism similar to the one implemented in the first sublayer of the encoder. On the decoder side, this multi-head mechanism receives the queries from the previous decoder sublayer and the keys and values from the output of the encoder. This allows the decoder to attend to all the words in the input sequence.
   </span></p>

   <p style="text-align: justify;"><span style="color:#000080;">
   
    The source-target attention is another multi-head attention that calculates the attention values between the features (embeddings) from the input sentence and the features from the output (yet partial) sentence.
   </span></p>

.. figure:: /Documentation/images/source_target.png
   :width: 700
   :align: center
   :alt: Alternative Text


5. Feed Forward
----------------

.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
   
   The third layer implements a fully connected feed-forward network, similar to the one implemented in the second sublayer of the encoder.
   </span></p>

.. note::
   urthermore, the three sublayers on the decoder side also have residual connections around them and are succeeded by a normalization layer.

   Positional encodings are also added to the input embeddings of the decoder in the same manner as previously explained for the encoder. 




6. Conclusion
---------------

.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
   The transformer architecture assumes no recurrence or convolution pattern when processing input data. As such, the transformer architecture is suitable for any sequence data. As long as we can express our input as sequence data, we can apply the same approach, including computer vision (sequences of image patches) and reinforcement learning (sequences of states, actions, and rewards).
   </span></p>  

   <p style="text-align: justify;"><span style="color:#000080;">
   
   In the case of the original transformer, the mission is to translate, and it uses the architecture to learn to enrich embedding vectors with relevant information for translation.
   </span></p>


.. figure:: /Documentation/images/conclusion.png
   :width: 600
   :align: center
   :alt: Alternative Text




.. admonition::  For more information

   .. container:: blue-box

      * `"transformers-encoder-decoder" <https://kikaben.com/transformers-encoder-decoder/#conclusion>`__
      
      * `"the-transformer_decoder_model" <https://machinelearningmastery.com/the-transformer-model/>`__

      * `"transformer-decoder" <https://www.scaler.com/topics/nlp/transformer-decoder/>`__

      

