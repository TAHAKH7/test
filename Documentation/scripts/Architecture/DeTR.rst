Detection Transformer
======================
5. Detection Transformer
-------------------------

DEtection TRansformer (DETR) model trained end-to-end on COCO 2017 object detection (118k annotated images). It was introduced in the paper `End-to-End Object Detection with Transformers <https://arxiv.org/abs/2005.12872>`__
 by Carion et al.

.. note::
  The  first released in this repository. `repository <https://github.com/facebookresearch/detr>`__


Explain the functioning and usage of the Detection Transformer (DeTR).

.. figure:: /Documentation/images/DTR.jpg
    :width: 400
    :align: center
    :alt: Alternative Text


.. figure:: /Documentation/images/prompt.png
   :width: 100%
   :alt: Alternative text for the image
   :name: logo

   
`paper Foundation Model Assisted Weakly Supervised Semantic Segmentation <https://arxiv.org/pdf/2312.03585v2.pdf>`__


Transformer Architecture
==========================

.. figure:: /Documentation/images/arch1.png
   :width: 400
   :align: center
   :alt: Alternative Text

The Transformer is a groundbreaking architecture in the field of natural language processing. In this context, we will explain the various aspects of this architecture.

    * **Introduction (Attention is All You Need)**

    .. note::  

      This introduction highlights the basics of the Transformer, as described in the paper "Attention is All You Need".
         
       `paper Attention is all you need <https://arxiv.org/pdf/1706.03762.pdf>`__ 

      

    * **Tokenization**
.. raw:: html

  <p style="text-align: justify;"><span style="color:#000080;">
   Tokenization is the process of converting text into tokens, the basic units on which the model operates.
  </span></p>
      

* **Embedding**
.. raw:: html


  <p style="text-align: justify;"><span style="color:#000080;">
  Embedding transforms tokens into dense vectors, which represent words numerically.
  </span></p>
      

* **Positional encoding**
.. raw:: html


  <p style="text-align: justify;"><span style="color:#000080;">
  Positional encoding adds information about the order of words in the sequence.
  </span></p>
      

* **Transformer block**
.. raw:: html


  <p style="text-align: justify;"><span style="color:#000080;">
  The Transformer block is the centerpiece of this architecture, comprising layers of attention and fully connected neural networks.
  </span></p>
      

* **Softmax**
.. raw:: html


  <p style="text-align: justify;"><span style="color:#000080;">
  Softmax is an activation function used to compute probability scores on the model's output.
  </span></p>
      

.. 
Visual Transformer (ViT)
==========================
.. note::
  paper:  
  `AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE  <https://arxiv.org/pdf/2010.11929v2.pdf>`__


Explain the functioning and usage of the Visual Transformer.

.. figure:: /Documentation/images/ViT.png
    :width: 400
    :align: center
    :alt: Alternative Text

.. _detection_transformer(DeTR):



.. image:: /Documentation/images/image2.gif
   :width: 700
   :align: center
   :alt: Alternative Text

.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">

   
    </span></p>

.. image:: /Documentation/images/segmentationgif.gif
   :width: 700
   :align: center
   :alt: Alternative Text


   Weakly Supervised Image Prompt Segmentation with Foundation Models documentation!
===================================================================================