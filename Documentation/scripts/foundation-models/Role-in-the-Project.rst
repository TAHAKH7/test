Groundingsam project implemention 
========================================

---------------------------------------------------------------------------------------------------------------------------------

.. raw:: html

    <p><span style="color:white;">'</p></span>
    <p style="text-align: justify;"><span style="color:#000080;"><i>  
   This document provides a detailed explanation of the code used to clone, install dependencies, and use the `GroundingSam` project to annotate and segment images.

   </i></span></p>

    <p><span style="color:white;">'</p></span>

Cloning the Repository
----------------------

.. code-block:: python

    !git clone https://github.com/SAAD1190/GroundingSam.git


.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">Clone the `GroundingSam` repository from GitHub.
    </i></span></p>  
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;"> This command downloads the project's source code from the GitHub repository into a directory named `GroundingSam`.
    </i></span></p>  
    <p><span style="color:white;">'</p></span>


Setting the Working Directory
-----------------------------

.. code-block:: python

    HOME = "/content/GroundingSam"

.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">Define a variable `HOME` for the path to the cloned directory.
    </i></span></p>  
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;"> `HOME` is used to simplify navigation and path management in the subsequent commands. This avoids the need to repeatedly specify the full path to the directory.
    </i></span></p>  
    <p><span style="color:white;">'</p></span>


 
Navigating to the Directory
----------------------------

.. code-block:: python

    %cd {HOME}



.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">Change the current working directory to the cloned directory.
    </i></span></p>  
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;"> 
    This command uses the `HOME` variable to navigate to the `GroundingSam` project directory, ensuring that all subsequent operations are executed in the correct directory.
    </i></span></p>  
    <p><span style="color:white;">'</p></span>



Installing Dependencies
------------------------

.. code-block:: python

    !bash dependencies.sh # Install the necessary dependencies (Ignore the pip dependency)


.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">Run the `dependencies.sh` script to install the necessary dependencies.
        </i></span></p>  
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;"> 
    This script installs all required dependencies for the project, except those to be installed via `pip`. It likely installs project-specific libraries or configures the environment.

        </i></span></p>  
    <p><span style="color:white;">'</p></span>


Creating Directories for Models and Annotations
-----------------------------------------------

.. code-block:: python

    !mkdir {HOME}/weights
    !mkdir {HOME}/annotations



.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">
    Create two directories, `weights` for model weights and `annotations` for annotations.
     
     </i></span></p>     
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;"> 
    These directories are created to organize the files needed for the project. The `weights` directory will contain the pre-trained model weights, while the `annotations` directory will store the generated annotation files.

        </i></span></p>  
    <p><span style="color:white;">'</p></span>



 
Downloading GroundingDINO Model Weights
---------------------------------------

.. code-block:: python

    %cd ./weights
    !wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth



.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">
    Navigate to the `weights` directory and download the GroundingDINO model weights.
     </i></span></p>     
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></i></span></p>
    <p style="text-align: justify;"><span style="color:red;"><i>   
    - cd ./weights: </span><span style="color:#000080;">Changes the current working directory to `weights`.
    </i></span></p> 
    <p style="text-align: justify;"><span style="color:red;"><i>   
    - wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth</span><span style="color:#000080;">  Downloads the GroundingDINO model weights from the provided URL. The `-q` option of `wget` makes the download silent, suppressing output.
    </i></span></p> 


    <p><span style="color:white;">'</p></span>


  
Returning to the Main Directory
-------------------------------

.. code-block:: python

    %cd {HOME}



.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">
    Return to the project's main directory after downloading the weights.
     </i></span></p>     
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">This command ensures continuity of operations in the main directory.
   </i></span></p>
    


    <p><span style="color:white;">'</p></span>


Installing Segment Anything via pip
-----------------------------------

.. code-block:: python

    !pip install 'git+https://github.com/facebookresearch/segment-anything.git'



.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">
   Install the `segment-anything` package from the GitHub repository.
      </i></span></p>     
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">
    This command installs the package directly via pip using the GitHub repository URL. This integrates the image segmentation functionalities from the Segment Anything project.

      </i></span></p>


    <p><span style="color:white;">'</p></span>



Downloading SAM Model Weights
-----------------------------

.. code-block:: python

    %cd ./weights
    !wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth


.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">
   Navigate again to the `weights` directory and download the SAM model weights.

        </i></span></p>     
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></i></span></p>
    <p style="text-align: justify;"><span style="color:red;"><i>   
    - cd ./weights: </span><span style="color:#000080;">
    Changes the current working directory to weights.
    </i></span></p> 
    <p style="text-align: justify;"><span style="color:red;"><i>   
    - wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth:</span><span style="color:#000080;">  
    Downloads the SAM model weights from the provided URL. The `-q` option of `wget` makes the download silent, suppressing output.
    </i></span></p> 


    <p><span style="color:white;">'</p></span>



Returning to the Main Directory (Again)
---------------------------------------

.. code-block:: python

    %cd {HOME}


.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">
   Return once more to the project's main directory.

      </i></span></p>     
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">
   Ensures that the following operations are performed in the main directory.

      </i></span></p>


    <p><span style="color:white;">'</p></span>


Importing and Initializing
--------------------------

.. code-block:: python

    from GroundingSam import *
    classes = ['piano', 'guitar', 'phone', 'hat']
    groundingsam = GroundingSam(classes=classes)



.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">
   Import the necessary modules from the `GroundingSam` project and initialize the `GroundingSam` object with a list of classes.

        </i></span></p>     
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></i></span></p>
    <p style="text-align: justify;"><span style="color:red;"><i>   
    - from GroundingSam import  </span><span style="color:#000080;">
    Imports all functions and classes from the `GroundingSam` module. This allows easy access to the module's functionalities.
 
    </i></span></p> 
    <p style="text-align: justify;"><span style="color:red;"><i>   
    - Classes: </span><span style="color:#000080;">  
   Defines a list of object classes to detect and annotate (in this case, 'piano', 'guitar', 'phone', 'hat').

       </i></span></p> 
    <p style="text-align: justify;"><span style="color:red;"><i>   
    - groundingsam = GroundingSam(classes=classes): </span><span style="color:#000080;">  
    Initializes the `GroundingSam` object with the specified classes. This object will be used for detection, annotation, and segmentation.

       </i></span></p> 

    <p><span style="color:white;">'</p></span>




Detection, Annotation, and Segmentation
---------------------------------------



.. code-block:: python

    from GroundingSam import *
    detections = groundingsam.get_detections()
    groundingsam.annotate_images()
    groundingsam.get_masks()

.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">
   Execute the main functions for detecting, annotating, and segmenting images.

        </i></span></p>     
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></i></span></p>
    <p style="text-align: justify;"><span style="color:red;"><i>   
    - detections = groundingsam.get_detections()  </span><span style="color:#000080;">
    Obtains object detections for the specified classes. This method uses the model weights to detect objects in images.
  </i></span></p> 
    <p style="text-align: justify;"><span style="color:red;"><i>   
    - groundingsam.annotate_images()</span><span style="color:#000080;">  
    Annotates the images based on the obtained detections. This method adds visual annotations (such as bounding boxes) to the images to indicate detected objects.

       </i></span></p> 
    <p style="text-align: justify;"><span style="color:red;"><i>   
    - groundingsam.get_masks() </span><span style="color:#000080;">  
     Generates segmentation masks for the detected objects. This method creates pixel-wise masks for each detected object, allowing for precise segmentation.

       </i></span></p> 

    <p><span style="color:white;">'</p></span>





.. code-block:: python

    from GroundingSam import *
    detections = groundingsam.get_detections()
    

.. figure:: /Documentation/images/foundation-models/imp/1.PNG
   :width:  700
   :align: center
   :alt: Alternative Text

.. raw:: html

    <p><span style="color:white;">'</p></span>


.. code-block:: python

    groundingsam.annotate_images()



.. figure:: /Documentation/images/foundation-models/imp/2.png
   :width:  700
   :align: center
   :alt: Alternative Text


.. raw:: html

    <p><span style="color:white;">'</p></span>


.. code-block:: python

    groundingsam.get_masks()


.. figure:: /Documentation/images/foundation-models/imp/3.png
   :width:  700
   :align: center
   :alt: Alternative Text




 
Summary
-------

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i>  
    This code implements and runs the `GroundingSam` project to annotate and segment images using pre-trained models. It includes cloning the repository, installing dependencies, creating directories, downloading model weights, and executing the functions for detection, annotation, and segmentation. The images are annotated with bounding boxes and segmentation masks for the specified classes.
       </i></span></p> 