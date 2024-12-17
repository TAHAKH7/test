Transformer Architecture
========================




Overview
-----------------


.. figure:: /Documentation/images/arch1.png
   :width: 400
   :align: center
   :alt: Alternative Text
.. raw:: html

   <p style="text-align: justify;"><span style="color:#00008B;">
   The Transformer is a groundbreaking architecture in the field of natural language processing. In this context, we will explain the various aspects of this architecture.

   </span></p>


   <p style="text-align: justify;"><span style="color:#00008B;">
   
   Introduced in 2017 and first presented in the groundbreaking paper "Attention is All You Need" (Vaswani et al. 2017), the Transformer model has been a revolutionary contribution to deep learning and, some argue, to computing as a whole. Born as a tool for automatic neural machine translation, it has proven to be of much greater scope, extending its applicability beyond natural language processing (NLP) and solidifying its position as a versatile and generalized neural network architecture.
   </span></p>
   <p style="text-align: justify;"><span style="color:#00008B;">
   
   In this comprehensive guide, we will dissect the Transformer model down to its fundamental structure, exploring in detail each key component, from its attention mechanism to its encoder-decoder architecture. Not stopping at the fundamental level, we will traverse the landscape of large language models that harness the power of Transformers, examining their unique design attributes and functionalities. Expanding further horizons, we will explore the applications of Transformer models beyond NLP and delve into the current challenges and potential future directions of this influential architecture. Additionally, a curated list of open-source implementations and additional resources will be provided for those interested in further exploration.
   </span></p>
   <p style="text-align: justify;"><span style="color:#00008B;">
   
   Without frills or fanfare, let's dive in!
   </span></p>


.. figure:: /Documentation/images/arch.png
   :width: 100%
   :align: center
   :alt: Alternative text for the image
   :name: Architecture




The purpose of Transformer networks
----------------------------------------



.. raw:: html

    <p style="text-align: justify;"><span style="color:#00008B;">
      In order to understand how Transformer networks work, it's important to understand the concept of attention. When translating a sentence from one language to another, rather than looking at each word individually, you consider the sentence as a whole and the context in which it is used. Some words are given more importance as they help to better understand the sentence. This is what we call attention.
    
     </span></p>

.. figure:: /Documentation/images/translation.png
   :width: 700
   :align: center
   :alt: Alternative text for the image



.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;">
    
    '

    </span></p>


.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;">
      
      Consider another example. Imagine that you are watching a movie and trying to understand a particular scene. Instead of focusing on a single frame, you pay attention to the sequence of frames and the actions of the characters in order to understand the overall story. This approach helps you understand the context.
    </span></p>

    <p style="text-align: justify;"><span style="color:#00008B;">
      
      In Transformer networks, attention is used to assign different levels of importance to different parts of the input sequence, which helps the model better understand and generate a coherent output sequence.
    </span></p>

    <p style="text-align: justify;"><span style="color:#00008B;">   
      
      The Transformer Network is powerful for tasks such as language understanding, due to its ability to capture long-range dependencies between elements that may be far apart from each other. This means that the relationships and dependencies between words in a sentence can be captured, even if they appear earlier or later in that sentence. This is important because the meaning of a word can depend on the words that appear before or after it.
     </span></p>


The Transformer Architecture
------------------------------------

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;">
      
      The Transformer architecture follows an encoder-decoder structure but does not rely on recurrence and convolutions in order to generate an output. 
   </span></p>



.. figure:: /Documentation/images/transf_arch.webp
   :width: 700
   :align: center
   :alt: Alternative text for the image

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;">
      
      In a nutshell, the task of the encoder, on the left half of the Transformer architecture, is to map an input sequence to a sequence of continuous representations, which is then fed into a decoder. 
    </span></p>

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;">
     
     The decoder, on the right half of the architecture, receives the output of the encoder together with the decoder output at the previous time step to generate an output sequence.
   </span></p>


.. note::
   At each step the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next.


Key Components
-------------------

.. figure:: /Documentation/images/key.jpg
   :width: 700
   :align: center
   :alt: Alternative text for the image


