Project Introduction
====================
------------------------------------------

image segmentation prompt
___________________________


.. figure:: /Documentation/images/scope/intro.jpg
   :width: 700
   :align: center
   :alt: Alternative text for the image

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;">

    This project aims to build a bridge (a connection) between users' text request and object detection inside an image.

   </span></p>
    <p style="text-align: justify;"><i>

    - <span style="color:blue;"> First input:</span><span style="color:#000080;"> Users' text request (query or prompt) about an object;

    </i></span></p>

    <p style="text-align: justify;"><i>

    - <span style="color:blue;"> Second input : </span><span style="color:#000080;"> The image;

    </i></span></p>

    <p style="text-align: justify;"><i>

    - <span style="color:blue;">Output : </span><span style="color:#000080;">The requested object, filtred and highlighted (segmented).
    </i></span></p>
    <p style="text-align: justify;">
    <span style="color:blue;"><strong>  For example: </span></strong>
    <span style="color:#000080;"><i>
    the user has an image of people playing in the park, and wants to filter out dogs in the picture.
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    in order to do so, the user inserts the picture and writes this query: "highlight dogs in the picture"

    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i> 

    The output would be a processed images where dogs are highlighted
    </i></span></p>




.. figure:: /Documentation/images/scope/exmpl.jpg
   :width: 700
   :align: center
   :alt: Alternative text for the image
   

.. raw:: html

    <p style="text-align: justify;">

    </p>

    <span style="color:blue;"><strong> How were we able to do that ?</strong></span>


    <p style="text-align: justify;"><span style="color:#000080;"><i>

    Building from scratch a model, that is trained on a dataset according to the field of interest.
    </i></span></p>

    <span style="color:blue;"><strong> What's new about the project ?</strong></span>

    <p style="text-align: justify;"><span style="color:#000080;"><i>

    Preparing an image dataset for training a model on segmentation is a time and energy consuming task, this process is done manually where one has to draw a contour on each object and label it.
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>

    The bridge, the connection or the model we are building from scratch uses FOUNDATION MODELS for training (look at like a human sitting on a computer, drawing contours and labeling each object on the image). This enable optimization of time and labor resources and open doors to the use of large-scale datasets for training and application purposes using flexible prompt.

    </i></span></p>


    <p style="text-align: justify;"><span style="color:#000080;"><i>
    
    This project goes way beyond the scope of detecting dogs in parks and may be used to perform object detection on any image in any field.

    </i></span></p>



    <span style="color:blue;"><strong>Project building strategy: </strong></span>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    Modular components
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    Manual implementation: Each component is implemented manually for pedagogical reasons
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    Build to last strategy : Simple, accessible documentation with practice examples
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    Accuracy-oriented: Replacing manually implemented components with imported frameworks for more accuracy

    </i></span></p>


.. raw:: html

    <p style="text-align: justify;">

    </p>


Documentation axes
_________________________

.. figure:: /Documentation/images/scope/3.jpg
   :width: 700
   :align: center
   :alt: Alternative text for the image

.. figure:: /Documentation/images/scope/4.jpg
   :width: 700
   :align: center
   :alt: Alternative text for the image
