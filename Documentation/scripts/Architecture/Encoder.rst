The Encoder
===========

.. figure:: /Documentation/images/encoder.webp
   :width:  700
   :align: center
   :alt: Alternative Text


1. Tokenizer 
-------------


.. figure:: /Documentation/images/token.jpg
   :width:  500
   :align: center
   :alt: Alternative Text

.. raw:: html

    <p style="text-align: justify;font-size: larger;"><span style="color:blue;">
   Tokenization:
   </span></p>
.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;">
      
   &#10003; Tokenization is the process of dividing a text into smaller units called "tokens."
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

   &#10003; These tokens can be individual words, sub-words, or even individual characters, depending on the desired level of granularity.
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

   &#10003; Each token is then converted into its corresponding numerical identifier from the model's vocabulary.
   </span></p>

.. raw:: html

    <p style="text-align: justify;font-size: larger;"><span style="color:blue;">
   Vocabulary Building:
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

   &#10003; To build the vocabulary, a set of the most common tokens in the language is typically selected.
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

   &#10003; The vocabulary is limited to a certain number of tokens for performance and efficiency reasons, usually tens of thousands of tokens.
   </span></p>

.. raw:: html

    <p style="text-align: justify;font-size: larger;"><span style="color:blue;">
   Token ID:
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

   &#10003; Each token is associated with a unique identifier called a "Token ID."
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">
   
   &#10003; These Token IDs serve as numerical references for each token in the model's vocabulary.
   </span></p>

.. raw:: html

    <p style="text-align: justify;font-size: larger;"><span style="color:blue;">
   Vocabulary Limitations:
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

   &#10003; Due to the limitation of vocabulary size, some words may not be present in the model's vocabulary.
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

   &#10003; In such cases, these words are usually split into sub-words or characters to represent them using the available tokens in the vocabulary.
   </span></p>


.. note::  

   More details in `Tokenization in Machine Learning Explained <https://vaclavkosar.com/ml/Tokenization-in-Machine-Learning-Explained>`__  


2. Input embedding
-------------------

.. figure:: /Documentation/images/input_embe.jpg
   :width:  500
   :align: center
   :alt: Alternative Text


.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
   Refers to the initial step of converting the discrete tokens of an input sequence into continuous vector representations. This process is essential for the model to work with the input data in a suitable format.
   </span></p>

.. raw:: html

    <p style="text-align: justify;font-size: larger;"><span style="color:blue;">
   Tokenization:
   </span></p>

.. figure:: /Documentation/images/input.jpg
   :width:  500
   :align: center
   :alt: Alternative Text

.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
   The input sequence, which could be a sequence of words, subwords, or characters, is first broken down into individual tokens. Each token typically represents a unit of meaning, like a word or a subword.
   </span></p>


    <p style="text-align: justify;font-size: larger;"><span style="color:blue;">
   Embedding Representation:
   </span></p>


   <p style="text-align: justify;"><span style="color:#000080;">
   Each token ID is associated with an embedding vector, where these vectors are initially randomly initialized. These vectors are of a fixed size, typically 512 dimensions.
   </span></p>


    <p style="text-align: justify;font-size: larger;"><span style="color:blue;">
   Vector Representation:
   </span></p>


   <p style="text-align: justify;"><span style="color:#000080;">
   These embedding vectors are arranged in columns, with each column representing a dimension of the embedding vector. This is different from the usual row-wise representation, where each row represents a token.
   </span></p>


    <p style="text-align: justify;font-size: larger;"><span style="color:blue;">
   Random Initialization:
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">  

    The values in the embedding vectors are initially set randomly. These values represent the initial state of the embeddings, and the Transformer model optimizes these values during training to better represent the semantics of the tokens.
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">  

    To sum up, the process involves tokenizing the input sentence, looking up each token in the vocabulary to retrieve its ID, then using this ID to obtain the corresponding embedding vector. These embedding vectors are represented in a column-wise format, with each column representing a dimension of the embedding vector. Initially, these vectors are randomly initialized, and the Transformer model learns to optimize them during training.
   </span></p>

.. note::  

   More details in `Transformer Positional Embeddings and Encodings <https://vaclavkosar.com/ml/transformer-positional-embeddings-and-encodings>`__  


3. Positional Encoding 
------------------------
.. figure:: /Documentation/images/position1.jpg
   :width:  600
   :align: center
   :alt: Alternative Text
.. raw:: html


.. figure:: /Documentation/images/position.jpg
   :width:  600
   :align: center
   :alt: Alternative Text
.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
   The significance of word position within a sentence is paramount. Depending on where a word is placed in a sentence, it can carry different meanings or implications. For instance, the word "not" might negate something if it appears in one part of the sentence, but it might have a different function elsewhere, such as merely continuing the speaker's thought without negating anything.
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">  

   This variation in word meaning based on position emphasizes the importance of "position embedding." While word embeddings represent the meaning of a word, position embeddings represent the position of the word within the sentence. However, it's important to note that position embeddings are usually calculated only once and are not subject to training like word embeddings.
   </span></p>

   <p style="text-align: justify;font-size: larger;"><span style="color:blue;">
  
   Mathematical Intuition
   </span></p>

   <p style="text-align: justify;"><span style="color:#000080;">
   
   The idea behind positional encoding is to add a fixed-size vector to the embeddings of the input tokens, and this vector is determined based on the position of the token in the sequence. The positional encoding is designed in such a way that it reflects the position of a token in the sequence space, allowing the model to discern the order of tokens during processing.
   </span></p>

.. figure:: /Documentation/images/position2.jpg
   :width:  600
   :align: center
   :alt: Alternative Text

.. raw:: html

   <p style="text-align: justify;">
   <ul class="circle-list"><span style="color:#006400;"><strong><li> d: </strong> </span><span style="color:#000080;"> The dimension of the embedding vector. This is the length or number of components in each vector that represents a token or position in the input sequence.</span>
   </ul> 

   
   <ul class="circle-list"><span style="color:#006400;"><strong><li> pos:</strong></span><span style="color:#000080;">  The position of the token in the sequence. It represents the index or order of the token in the input sequence.</span>
   </ul> 


   <ul class="circle-list"><span style="color:#006400;"><strong><li> i:</strong></span><span style="color:#000080;">  he position along the dimension of the embedding vector. For each dimension i, there is a corresponding sine term (for even indices) and cosine term (for odd indices) in the formula.</span>
   </ul> 
   </p>


.. note::  

   More details in `Transformer Positional Embeddings and Encodings <https://vaclavkosar.com/ml/transformer-positional-embeddings-and-encodings>`__  


4. self Attention
-------------------

.. figure:: /Documentation/images/self.png
   :width:  500
   :align: center
   :alt: Alternative Text

.. note:: 

   self-attention (sometimes KQV-attention) layer is central mechanism in transformer architecture introduced in `Attention Is All You Need paper<https://arxiv.org/pdf/1706.03762.pdf>`__ 

.. figure:: /Documentation/images/cal.jpg
   :width:  700
   :align: center
   :alt: Alternative Text


.. raw:: html

   <p style="text-align: justify;">
   <span style="color:#000080;">Self-Attention compares all input sequence members with each other, and modifies the corresponding output sequence positions. In other words, self-attention layer differentiably key-value searches the input sequence for each inputs, and adds results to the output sequence.</span>
   
   </p>
   <span style="color:blue;font-size: larger;">Key</span>, <span style="color:blue;font-size: larger;">Query</span>, and <span style="color:blue;font-size: larger;">Value:</span>

   <p style="text-align: justify;">
   <span style="color:#000080;">Each word in the input sequence is associated with three vectors: </span><span style="color:red;"><strong>Key (K)</strong></span>,<span style="color:red;"><strong> Query (Q)</strong></span>,<span style="color:#000080;"> and</span> <span style="color:red;"><strong>Value (V)</strong></span><span style="color:#000080;">. These vectors are learned parameters for each word. Vectors are used to compute attention scores, determining how much focus should be given to other words in the sequence.</span>
   
   </p>
   <span style="color:blue;font-size: larger;">Attention Scores:</span>

   <p style="text-align: justify;">
   <span style="color:#000080;"> For each word, the attention score with respect to other words is calculated by taking the dot product of the Query vector of the current word with the Key vectors of all other words. The scores are then scaled and passed through a softmax function to obtain a probability distribution, ensuring that the weights add up to 1.</span>
   
   </p>
   <span style="color:blue; font-size: larger;">Weighted Sum:</span>

   <p style="text-align: justify;">

   <span style="color:#000080;">The attention scores obtained for each word are used to calculate a weighted sum of the corresponding Value vectors. This weighted sum represents the importance of each word in the context of the current word, capturing the dependencies in the sequence.</span>

   </p>

.. note::  

   More details in `paper Attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`__  : dot-product is “scaled”, residual connection, layer normalization


5. Multi-Head Attention
------------------------

.. figure:: /Documentation/images/multi.jpg
   :width:  700
   :align: center
   :alt: Alternative Text

.. raw:: html
      
   <p style="text-align: justify;"><span style="color:#000080;">
   &#10003;  In multi-head attention, the attention mechanism is applied multiple times in parallel, with each instance referred to as a "head." 
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">
   &#10003;  For each head, three learnable linear projections (matrices) are applied to the input sequence to obtain separate projections for the query (Q), key (K), and value (V) vectors.
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">
   &#10003;  The attention mechanism is then applied independently to each of these query, key, and value projections. The resulting outputs from all heads are concatenated and linearly transformed to produce the final multi-head attention output.
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">
   &#10003;  The use of multiple heads allows the model to focus on different parts of the input sequence for different aspects or patterns, providing more flexibility and expressiveness.
   </span></p>
.. figure:: /Documentation/images/multical.jpg
   :width: 500
   :height: 100
   :align: center
   :alt: Alternative Text



.. figure:: /Documentation/images/multii.png
   :width:  400
   :align: center
   :alt: Alternative Text


.. figure:: /Documentation/images/multi_head.png
   :width:  400
   :align: center
   :alt: Alternative Text


.. raw:: html
      
   <p style="text-align: justify;"><span style="color:#000080;">

   In practice, to enhance the capability of attention mechanisms to capture dependencies of various ranges within a sequence, a technique called multi-head attention is employed. Instead of a single attention pooling operation, multi-head attention utilizes independently learned linear projections for queries, keys, and values. These projected queries, keys, and values are then simultaneously processed through attention pooling in parallel. Subsequently, the outputs from each attention pooling operation, referred to as heads, are concatenated and transformed using another learned linear projection to generate the final output. Multi-head attention allows the model to combine knowledge from different behaviors of the attention mechanism, thereby improving its ability to capture dependencies across different ranges within a sequence. This approach is illustrated in Figure  where fully connected layers are employed for learnable linear transformations.
   </span></p>


.. figure:: /Documentation/images/multi_head_3.jpg
   :width:  400
   :align: center
   :alt: Alternative Text




6. Add and norm - Norm
------------------------


.. figure:: /Documentation/images/add_norm.jpg
   :width:  400
   :align: center
   :alt: Alternative Text


.. raw:: html
      
   <p style="text-align: justify;"><span style="color:#000080;">

   Following the addition operation, layer normalization is applied to normalize the combined result. Layer normalization normalizes the activations across the feature dimension (e.g., the dimension of the embedding vectors) for each position in the sequence.
   </span></p>
   
   <p style="text-align: justify;"><span style="color:#000080;">

   The layer normalization operation is typically expressed as  LayerNorm(Output) 
   </span></p>
   
   <p style="text-align: justify;"><span style="color:#000080;">   
   where is a learnable normalization function.
   </span></p>

   <p style="text-align: justify;"><span style="color:#000080;">
   This normalization step helps stabilize the training process by ensuring that the model's inputs and outputs have similar magnitudes, which can be beneficial for convergence and generalization.
   </span></p>

7. Feed Forward
----------------
.. raw:: html
      
   <p style="text-align: justify;"><span style="color:#000080;">

   A specific type of neural network layer that is used within each encoder and decoder block. The feedforward layer is responsible for processing the information captured by the self-attention mechanism in the model.
   </span></p>


.. figure:: /Documentation/images/feedforward.jpg
   :width:  400
   :align: center
   :alt: Alternative Text


.. raw:: html
      
   
   <span style="color:blue;font-size: larger;">
   Input
   </span>
   <p style="text-align: justify;"><span style="color:#000080;">

   The output from the self-attention mechanism-Add & Norm Sublayer
   </span></p>
   <span style="color:blue;font-size: larger;">
   Linear Transformation
   </span>
   <p style="text-align: justify;"><span style="color:#000080;">

   Input is passed through a linear transformation
   </span></p>
   <span style="color:blue;font-size: larger;">
   Activation Function
   </span>
   <p style="text-align: justify;"><span style="color:#000080;">

   Application of a non-linear activation function, typically a rectified linear unit (ReLU). 

   To introduces non-linearity to the model, allowing it to capture more complex patterns in the data.
   </span></p>
   <span style="color:blue;font-size: larger;">
   Second Linear Transformation
   </span>
   <p style="text-align: justify;"><span style="color:#000080;">

   The output from the activation function undergoes another linear transformation with a different weight matrix and bias term. This step further refines the information.
   </span></p>
   <span style="color:blue;font-size: larger;">
   Output
   </span>
   <p style="text-align: justify;"><span style="color:#000080;">

   The final result is the output of the feedforward layer, and it is passed on to subsequent layers in the model.
   </span></p>


**Mathematical Intuition**


.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">

   Mathematically, if X represents the input, the feedforward layer can be expressed as:
   </span></p>


.. figure:: /Documentation/images/mathsfeed.jpg
   :width: 500
   :height: 100
   :align: center
   :alt: Alternative Text


.. raw:: html

   <p style="text-align: justify;">
   <span style="color:red;font-size: larger;">W1​, b1​ </span>:<span style="color:#000080;"> the weights and bias for the first linear transformation.</span>
   </p>
   <p style="text-align: justify;">
   <span style="color:red;font-size: larger;">W2​, b2​ </span>: <span style="color:#000080;">the weights and bias for the second linear transformation.</span>
   </p>
   <p style="text-align: justify;">
   <span style="color:red;font-size: larger;">ReLU</span> : <span style="color:#000080;">the rectified linear unit activation function.</span>
   </p>

.. note::

   .. raw:: html

      <p style="text-align: justify;">
      The feedforward layer plays a crucial role in capturing complex patterns and relationships in the input data, allowing the model to learn and represent hierarchical features effectively.
      </p>

8. Residual Connections
-------------------------

.. raw:: html

   <p style="text-align: justify;">
   Another point is that the encoder block uses residual connections, which is simply an element-wise addition:
   </p>

.. note::   

   .. raw:: html

      <p style="text-align: justify;">
      Sublayer is either multi-head attention or point-wise feed-forward network.
      </p>

   
.. figure:: /Documentation/images/residual.jpg
   :width: 500
   :height: 100
   :align: center
   :alt: Alternative Text


.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
   
   Residual connections carry over the previous embeddings to the subsequent layers. As such, the encoder blocks enrich the embedding vectors with additional information obtained from the multi-head self-attention calculations and position-wise feed-forward networks.
   </span></p>


9. Conclusion
---------------


.. figure:: /Documentation/images/conc_encoder.png
   :width: 700
   :height: 500
   :align: center
   :alt: Alternative Text




.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
      
   The document provides a detailed overview of <strong>the Transformer model encoder</strong> and the different stages and mechanisms used in data processing. 
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">
   
   First, the <strong>tokenization</strong> process is explained, which involves dividing a text into smaller units called "tokens". These tokens are then converted into numerical identifiers that correspond to the model's vocabulary. 
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

   Next, <strong>input embedding</strong> is described, where tokens are converted into continuous vectors for processing by the model. These vectors are randomly initialized and optimized during training to better represent the meaning of the tokens. 
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

   <strong>Positional encoding </strong>is discussed to account for the order of tokens in the sequence, which is crucial for capturing dependencies between words. 
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

   <strong>The self-attention mechanism</strong> is introduced as a method for calculating attention weights between tokens, allowing the model to focus on different parts of the sequence depending on the context. 
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">
   Next, <strong>multi-head attention </strong>is explained, which is an extension of self-attention that allows the model to combine knowledge from different attention heads to capture dependencies across different ranges within a sequence. 
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

   <strong>The addition and normalization layer</strong>, as well as the feed-forward layer are discussed to improve stability and the model's ability to learn complex hierarchical representations. 
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

   Finally, <strong>residual connections</strong> are introduced to facilitate the flow of information through the encoder's layers. 
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">
   In<span style="color:blue;"> conclusion</span>, the document provides a comprehensive understanding of the internal workings of the Transformer encoder and highlights the importance of the various stages and mechanisms in effective data processing.
   </span></p>

.. note::  
    * You can view more by clicking the  `"Transformer’s Encoder" <https://kikaben.com/transformers-encoder-decoder/#conclusion>`__ 


10. BIBLIOGRAPHIC
-------------------


.. admonition::  For more information

   .. container:: blue-box


      * `"what is a transformer" <https://medium.com/@francescofranco_39234/what-is-a-transformer-part-2-a2694745774a>`__

      * `"Transformers-Encoder-Decoder" vidéo YouTube <https://www.youtube.com/watch?v=4Bdc55j80l8>`__ 

      * `"transformers-important-architecture" <https://cash-ai.news/2024/03/01/what-are-transformers-important-architecture-details-by-akshitha-kumbam-mar-2024/>`__
   
      * `"transformers-introduction" <https://pylessons.com/transformers-introduction>`__  
   
      * `"Attention is all you need" vidéo YouTube <https://www.youtube.com/watch?v=sznZ78HquPc>`__ 
   
      * `Mécanismes d'attention <https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html>`__ 



