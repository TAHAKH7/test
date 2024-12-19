GroundingSam with customized weights
========================================

---------------------------------------------------------------------------------------------------------------------------------

.. raw:: html

    <p><span style="color:white;">'</p></span>
    <p style="text-align: justify;"><span style="color:#000080;"><i>  
    GroundingSam output after importing finetunned GroundingDINO weight and SAM original weights 
   </i></span></p>

    <p><span style="color:white;">'</p></span>

Clone the Repository
----------------------

.. code-block:: python

    !git clone https://github.com/SAAD1190/GroundingSam.git


.. raw:: html
 
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">This step clones the GroundingSam repository to access its codebase, which includes pre-built functionality for integrating GroundingDINO with the Segment Anything Model (SAM)
    </i></span></p>  
    <p><span style="color:white;">'</p></span>


Install Dependencies
-----------------------------

.. code-block:: python

    HOME = "/content/GroundingSam"
    %cd {HOME}
    !bash dependencies.sh  # Install the necessary dependencies



.. raw:: html
  
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">Moves to the repository directory and runs a script (dependencies.sh) that installs required Python libraries and dependencies for the project
    </i></span></p>  
    <p><span style="color:white;">'</p></span>
 
Create Folders for Model Weights and Annotations
----------------------------

.. code-block:: python

    !mkdir {HOME}/weights
    !mkdir {HOME}/annotations


.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">Creates directories to store downloaded model weights and annotations generated during the detection and segmentation tasks
    </i></span></p>  
    <p><span style="color:white;">'</p></span>


Download Custom Model Weights
------------------------

.. code-block:: python

    %cd ./weights
    from google.colab import drive
    drive.mount('/content/drive')

    # Install gdown if not already installed
    !pip install gdown

    # Download the custom weights from Google Drive
    file_id = "1ovh5uuY2YdqKadh_Niy5FHAX8YXTvGOQ"
    output_file = "downloaded_file"

    !gdown --id {file_id} -O {output_file}
    %cd {HOME}




.. raw:: html
  
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">Mounts Google Drive to access a shared file containing custom-trained model weights. These weights are downloaded and saved into the weights folder using the gdown library
        </i></span></p>  
    <p><span style="color:white;">'</p></span>


Install the Segment Anything Model (SAM)
-----------------------------------------------

.. code-block:: python

    !pip install 'git+https://github.com/facebookresearch/segment-anything.git'


.. raw:: html

     </i></span></p>     
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;"> 
    Installs the SAM library directly from its GitHub repository to enable segmentation functionality
        </i></span></p>  
    <p><span style="color:white;">'</p></span>



 
Download SAM Pre-trained Weights
---------------------------------------

.. code-block:: python

    %cd ./weights
    !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
    %cd {HOME}

.. raw:: html
    
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></i></span></p>Downloads pre-trained weights for SAM from Facebook's public file repository and saves them into the weights folder
        </i></span></p> 
    <p><span style="color:white;">'</p></span>


Import and Initialize the GroundingSam Library
-------------------------------

.. code-block:: python

   from GroundingSam import *
    classes = ['crack']
    groundingsam = GroundingSam(classes=classes)


.. raw:: html
   
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">Imports the GroundingSam library and initializes it with the class names (e.g., crack) that will be used for object detection and segmentation tasks.
   </i></span></p>
    


    <p><span style="color:white;">'</p></span>


Run Detection
-----------------------------------

.. code-block:: python

   detections = groundingsam.get_detections()




.. raw:: html
   
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">Generates object detections using the GroundingDINO model integrated into the GroundingSam library.
      </i></span></p>
    <p><span style="color:white;">'</p></span>



Annotate Images
-----------------------------

.. code-block:: python

    groundingsam.annotate_images()

.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">
    Annotates the detected objects on the images. This step overlays bounding boxes or masks on the images based on the detected objects.
    <p><span style="color:white;">'</p></span>

.. figure:: /Documentation/images/annotate_cast.png
   :width:  700
   :align: center
   :alt: Alternative Text

.. raw:: html

    <p><span style="color:white;">'</p></span>

Generate Segmentation Masks
---------------------------------------

.. code-block:: python

    groundingsam.get_masks()

.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">
    Generates segmentation masks for the detected objects using the SAM model. These masks are used for detailed segmentation of the objects within the images.
    <p><span style="color:white;">'</p></span>
    
.. figure:: /Documentation/images/mask_cast.png
   :width:  700
   :align: center
   :alt: Alternative Text

.. raw:: html

    <p><span style="color:white;">'</p></span>
