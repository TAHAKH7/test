Visual Transformer (ViT)
========================


.. image:: /Documentation/images/Building-Blocks/ViT/vit.gif
   :width: 700
   :align: center
   :alt: Alternative Text



1. introduction
-----------------





.. raw:: html
      
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    Vision Transformers are type of deep learning model and they are designed for computer vision tasks, they are inspired by the success of Transformer models in natural language processing. Traditionally computer used a technique called convolutional neural networks for computer vision tasks but now the vision Transformers are newer approach that gained a lot of attention.
    </i></span></p>


.. figure:: /Documentation/images/Building-Blocks/ViT/vit1.jpg
    :width: 700
    :align: center
    :alt: Alternative Text

.. raw:: html
      
    <p style="text-align: justify;"><span style="color:#000080;">
    
    </span></p>


For more Understanding Vision Transformers

.. admonition:: link for more information

   .. container:: custom-red-box

      * `Vision Transformers <https://paperswithcode.com/method/vision-transformer>`__



.. raw:: html
      
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    The vision Transformer architecture. In this example, an image is split into nine patches. A special “<cls>” token and the nine flattened image patches are transformed via patch embedding and <strong><i> n </i></strong>Transformer encoder blocks into ten representations, respectively. The “<cls>” representation is further transformed into the output label.
    
    </i></span></p>


.. figure:: /Documentation/images/Building-Blocks/ViT/Patch_embeddingint.jpg
    :width: 700
    :align: center
    :alt: Alternative Text


.. raw:: html
      
    <p style="text-align: justify;"><span style="color:#000080;">
    
    </span></p>

For more Understanding Vision Transformers

.. admonition:: link for more information

   .. container:: custom-red-box

      * `The vision Transformer architecture <https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html#fig-vit>`__


.. note::

    Let’s explain how the vision transformer works step by step with an example.


2. Vision Transformers example
-------------------------------


* **example**


.. figure:: /Documentation/images/Building-Blocks/ViT/Patch_embedding1.jpg
    :align: center
    :alt: Alternative Text

    
.. raw:: html
      

    <p style="text-align: justify;">
    
    </p>

      
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    Patch = Square region of the image (4 patches in the example above)
    </i></span></p>
      
    <p style="text-align: justify;">
    
    </p>


.. figure:: /Documentation/images/Building-Blocks/ViT/Patch_embedding2.jpg
    :align: center
    :alt: Alternative Text


.. raw:: html
      
    <p style="text-align: justify;">
    
    </p>

      
    <p style="text-align: justify;"><i>

    In the example, we have an<span style="color:#000080;"> image size 224x224</span> pixels <span style="color:#000080;">the patch size is 16x16 pixels</span>, so we divide the width and height of the image by the patch size to get the total number of patches. In this case we would end up with <span style="color:#000080;">196 patches </span>covering the entire image.
    </i></p>

    <p style="text-align: justify;"><span style="color:#000080;">

    <i>An additional concept is the stride, meaning how many pixels the sliding moves each time. The stride used in the original paper is also 16.  So there would be no overlap between the patches because stride is equal to the patch size.
    </i></span></p>

      
    <p style="text-align: justify;">
    
    </p>


.. figure:: /Documentation/images/Building-Blocks/ViT/Patch_embedding3.jpg
    :align: center
    :alt: Alternative Text


.. raw:: html
      
    <p style="text-align: justify;">
    
    </p>

    <p style="text-align: justify;"><i>

     Next step is to flatten those patches from 2D Vector to a 1D Vector. Each patch is treated as a separate input token.
    </i></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>

    Now let's understand the tokenization and how image patches are flattened using the example above. In order for a computer to understand and analyze the image we need to break it down into smaller parts, these smaller parts are called image patches. Each patch contains a small portion of the image (the information). For example, one patch may contain hair, another patch may have nose or eye. Let's focus on one patch for a better understanding of the concept.
    </i></span></p>


      
    <p style="text-align: justify;">
    
    </p>


.. figure:: /Documentation/images/Building-Blocks/ViT/Patch_embedding4.jpg
    :align: center
    :alt: Alternative Text


.. raw:: html
      
    <p style="text-align: justify;">
    
    </p>

    <p style="text-align: justify;"><span style="color:#000080;"><i>  

    Suppose we focus on one patch, the one containing the right eye. (it's like a mini picture of the <strong> puzzle</strong>). Instead of treating it like a picture, we want is we want that the computer to process it as a sequence of smaller elements called tokens. To do this we need to further <strong> flatten</strong> the patch. In this case, the patch size is a 16x16, which makes a sequence of (16x16=256) 256 tokens 
    </i> </span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i>  
    
    Now that we converted the image patch into a sequence of tokens. These tokenized patches will be served as<strong> the input to the Transformer model.</strong>
    </i> </span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>  

    By breaking down the image into smaller patches and by converting them into a sequence of tokens the Transformer model can process and understand the different parts of the image.
    </i> </span></p>
    <p style="text-align: justify;">
    
    </p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>  

    Unlike the transformer model, Vision Transformer is an encoder only Transformer there is no</i> </span><span style="color:red;"> decoder</span>.
    </p>
    <p style="text-align: justify;">
    
    </p>


.. figure:: /Documentation/images/Building-Blocks/ViT/Patch_embedding5.jpg
    :align: center
    :alt: Alternative Text


.. raw:: html
      
    <p style="text-align: justify;">
    
    </p>

    <p style="text-align: justify;"><span style="color:#000080;"><i>  

    So, let's explain Vision Transformer layers with the same example and more simplified dimensions. Starting with an image of size 32x32 pixels and four patches of size 16x16 and a stride of 16 to prevent overlapping. First step is to flatten patches. So, we take the 2D patch and then we flatten it into a one-dimensional vector of 256 tokens (16x16=256), each token represents a specific part of the patch like a pixel 
    </i></span></p>
    <p style="text-align: justify;">
    
    </p>
.. figure:: /Documentation/images/Building-Blocks/ViT/Patch_embedding6.jpg
    :align: center
    :alt: Alternative Text


.. raw:: html
      
    <p style="text-align: justify;">
    
    </p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>  

    Next, we have linear projection working by transforming each 1D Vectorinto a lower dimensional Vector while preserving the relationships andimportant features.

    </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i>  

    The linear projection involves two main steps; first is weight matrix multiplication and the second one is bias addition. This is like the convolutional neural network when we multiply weights with the input and then we add bias. 
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>  

    The same thing is happening in linear projection, so this involves multiplying each element of the flattened sequence by a weight and adding a bias term the weights and biases are learned during the training process.
    </i></span></p>

    <p style="text-align: justify;">

    </span></p>

.. figure:: /Documentation/images/Building-Blocks/ViT/Patch_embedding7.jpg
    :align: center
    :alt: Alternative Text


.. raw:: html
      
    <p style="text-align: justify;">
    
    </p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>  

    The result of these two steps is a transformed Vector of lower dimensionality. The meaning of a vector of lower dimensionality refers to a vector that has fewer elements compared to the original or it represents a reduction in the number of dimensions or features used to represent a particular object (patch).
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>  
    
    Note that lower dimensional vectors require less memory and less computational resources making the process faster and more efficient. The other point is by reducing the dimensionality we can extract essential features and capture the most important information while discarding the less significant details and eliminating noise and irrelevant variations in the data preparation process.
    </i></span></p>

    <p style="text-align: justify;">

    </span></p>


.. figure:: /Documentation/images/Building-Blocks/ViT/Patch_embedding8.jpg
    :align: center
    :alt: Alternative Text



.. raw:: html
      
    <p style="text-align: justify;">
    
    </p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>  

     Next step is, positional embedding is added to each flattened image patch indicating each patch location in the image. Because, when we feed data to Transformer, we feed all the data at once, so Transformer doesn’t know the right order of the patches in the original image (which patch is first and which path should be the second part of the image). So, with positional embedding we provide the position information to the Transformer 
    </i></span></p>

    <p style="text-align: justify;">

    </span></p>


.. figure:: /Documentation/images/Building-Blocks/ViT/Patch_embedding9.jpg
    :align: center
    :alt: Alternative Text



.. raw:: html
      
    <p style="text-align: justify;">
    
    </p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    
    &#10003; The vector obtained after adding positional embedding is fed to the next layers of the Vision Transformer for further processing. The first layer of the encoder is self-attention layer.
    </i>  </span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>

    &#10003; Self-attention allows each patch to attend and gather information from other patches, it captures dependencies between the patches and enables the model to consider the global context.
    </i>  </span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>

    &#10003; After self-attention layer we have feed forward Network and the output of each patch is passed through a feed forward neural network, this helps capture complex non-linear relationships within the patches.
    </i>  </span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>

    &#10003; The final layer, MLP layer is a classification that maps the output of the transformer into the desired output format (image classification)
    </i>  </span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>

    &#10003; Instead of the decoder there is just the extra linear layer for final classification which is called MLP head. The absence of decoder is one of the key differences between the vision Transformer and the traditional Transformer architecture used in natural language processing where we perform translations or text Generations. In this context, we need decoder it is used to generate output sequences based on the Learned representations.
    </i>  </span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>    
    &#10003; However, in computer vision tasks such as image classification or object detection the primary goal of vision Transformer is to extract meaningful feature and to understand their spatial relationships within the image so the encoder in a vision Transformer performs this task by leveraging self-attention mechanism mechanisms to capture both local and Global dependencies between image and patches.
    </i>  </span></p>

