Groundingsam project implemention 
========================================

---------------------------------------------------------------------------------------------------------------------------------

.. raw:: html

    <p><span style="color:white;">'</p></span>
    <p style="text-align: justify;"><span style="color:#000080;"><i>  
   After implementing and running the GroundingSam model in order to annotate and segment objects (general approach), in this part of the project we will finetunned it to a specific dataset (industrial product data containing defects)

   </i></span></p>

    <p><span style="color:white;">'</p></span>

Clone the Repository
----------------------

.. code-block:: python

    !git clone "https://github.com/longzw1997/Open-GroundingDino"


.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">The repository Open-GroundingDino is cloned to access the framework for fine-tuning.
    </i></span></p>  
    


Install Requirements
-----------------------------

.. code-block:: python

    %cd Open-GroundingDino
    !pip install -r requirements.txt
    %cd models/GroundingDINO/ops
    !python setup.py install
    !python test.py
    %cd /content


.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">Navigate to the repository and install the required Python packages and Build the operations for GroundingDINO
    </i></span></p>  

 
Dataset Preparation
----------------------------

.. code-block:: python

    import os
    os.makedirs("/content/data", exist_ok=True)
    !unzip /content/1127_crack_pipe_wall.v2i.coco.zip -d /content/data


    import matplotlib.pyplot as plt
    import random
    from PIL import Image

    # Path to the folder containing the images
    folder_path = '/content/data/train'

    # Randomly select 16 images from the folder
    all_files = os.listdir(folder_path)
    image_files = [file for file in all_files if file.lower().endswith(('png', 'jpg', 'jpeg'))]
    selected_images = random.sample(image_files, 16)

    # Plot each selected image
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, image_file in zip(axes.flatten(), selected_images):
        img = Image.open(os.path.join(folder_path, image_file))
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()




.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">The dataset in COCO format is downloaded, unzipped, and organized.
    Visualize the dataset by displaying random images.
    </i></span></p>  



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