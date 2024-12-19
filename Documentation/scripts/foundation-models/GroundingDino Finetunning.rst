GroundingDINO Finetunning 
========================================

---------------------------------------------------------------------------------------------------------------------------------

.. raw:: html

    <p><span style="color:white;">'</p></span>
    <p style="text-align: justify;"><span style="color:#000080;"><i>  
   After implementing and running the GroundingSam model in order to annotate and segment objects (general approach), in this part of the project we will finetunne GroundingDINO to a specific dataset (industrial product data containing defects)

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
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">This step clones the Open-GroundingDino repository, which contains the necessary codebase and scripts for fine-tuning the GroundingDino model.
    </i></span></p>  
    <p><span style="color:white;">'</p></span>


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
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">Change the working directory to the cloned repository and installs all required Python dependencies listed in the requirements.txt file.
    Compile the custom operations used by GroundingDino for model training and inference.
    </i></span></p>  
    <p><span style="color:white;">'</p></span>
 
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

.. figure:: /Documentation/images/output1.png
   :width:  700
   :align: center
   :alt: Alternative Text

.. raw:: html

    <p><span style="color:white;">'</p></span>



.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">The dataset in COCO format is downloaded, unzipped, and organized.
    Visualize the dataset by displaying random images.
    </i></span></p>  
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">Creates a directory for storing the dataset and unzips the provided COCO-format dataset into this directory.
    Randomly selects 16 images from the dataset and visualizes them in a grid to ensure the data is correctly loaded.
    </i></span></p>  
    <p><span style="color:white;">'</p></span>


Convert Dataset to Custom Format
------------------------

.. code-block:: python

    import re

    # Modify `coco2odvg.py` to map dataset-specific IDs and labels
    file_path = 'Open-GroundingDino/tools/coco2odvg.py'
    new_id_map = '{0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7}'
    new_ori_map = '{"1": "fish", "2": "jellyfish", "3": "penguins", "4": "sharks", "5": "puffins", "6": "stingrays", "7": "starfish"}'

    with open(file_path, 'r') as file:
        content = file.read()
    content = re.sub(r'id_map\s*=\s*\{[^\}]*\}', f'id_map = {new_id_map}', content)
    content = re.sub(r'ori_map\s*=\s*\{[^\}]*\}', f'ori_map = {new_ori_map}', content)
    with open(file_path, 'w') as file:
        file.write(content)
    


    !pip install jsonlines
    !python /content/Open-GroundingDino/tools/coco2odvg.py \
    --input "/content/data/train/_annotations.coco.json" \
    --output "/content/input_params/train.jsonl"



.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">Modify Dataset Mappings and Run the Conversion.
        </i></span></p>  
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">Updates the coco2odvg.py script to correctly map dataset-specific IDs and labels for conversion.
    Converts the COCO-format annotations into the custom odvg format required for GroundingDino.
        </i></span></p>  
    <p><span style="color:white;">'</p></span>


Modify Configuration Files
-----------------------------------------------

.. code-block:: python

    def modify_file(file_path):
    label_list_content = 'label_list = ["fish","jellyfish","penguins","sharks","puffins","stingrays","starfish"]\n'
    with open(file_path, 'r') as file:
        content = file.read()
    content = re.sub(r'use_coco_eval\s*=\s*True', 'use_coco_eval = False', content)
    content = re.sub(r'use_coco_eval\s*=\s*False', r'use_coco_eval = False\n\n' + label_list_content, content, count=1)
    with open(file_path, 'w') as file:
        file.write(content)

    modify_file('/content/Open-GroundingDino/config/cfg_coco.py')
    modify_file('/content/Open-GroundingDino/config/cfg_odvg.py')




.. raw:: html

     </i></span></p>     
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;"> 
    Disables COCO evaluation and adds the dataset-specific labels in the configuration files for training.
        </i></span></p>  
    <p><span style="color:white;">'</p></span>



 
Download Required Models
---------------------------------------

.. code-block:: python

    !wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained("/content/bert")
    model.save_pretrained("/content/bert")




.. raw:: html
    
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></i></span></p>Downloads the pre-trained weights for the GroundingDino model.
    Saves the tokenizer and model for BERT, which will be used during training.
        </i></span></p> 
    <p><span style="color:white;">'</p></span>


Prepare Training Script  
-------------------------------

.. code-block:: python

    train_script_content = """
    CFG=$1
    DATASETS=$2
    OUTPUT_DIR=$3

    export CUDA_VISIBLE_DEVICES=0

    python main.py \\
        --config_file ${CFG} \\
        --datasets ${DATASETS} \\
        --output_dir ${OUTPUT_DIR} \\
        --pretrain_model_path /content/groundingdino_swint_ogc.pth \\
        --options text_encoder_type="/content/bert"
    """
    with open('/content/Open-GroundingDino/train_dist.sh', 'w') as file:
        file.write(train_script_content)




.. raw:: html
   
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">Creates a custom training script to run on a single GPU, specifying paths for weights, datasets, and output directories.
   </i></span></p>
    


    <p><span style="color:white;">'</p></span>


Train the Model
-----------------------------------

.. code-block:: python

    %cd /content/Open-GroundingDino
    GPU_NUM=1
    CFG="/content/Open-GroundingDino/config/cfg_odvg.py"
    DATASETS="/content/Open-GroundingDino/config/datasets_mixed_odvg.json"
    OUTPUT_DIR="/content/output"
    !chmod +x train_dist.sh
    !bash train_dist.sh $CFG $DATASETS $OUTPUT_DIR



.. raw:: html
   
    <p style="text-align: justify;"><span style="color:blue;"><i> 
    - <strong>Explanation:</strong></span><span style="color:#000080;">Starts the training process using the fine-tuned configurations and dataset.
      </i></span></p>
    <p><span style="color:white;">'</p></span>



Perform Inference
-----------------------------

.. code-block:: python

    image_dir = "/content/data/test"
    output_dir = "pred_images"
    config_path = "/content/Open-GroundingDino/tools/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "/content/output/checkpoint0014.pth"
    text_prompts = "crack"

    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.png', '.jpg')):
            image_path = os.path.join(image_dir, image_file)
            command = [
                "python", "/content/Open-GroundingDino/tools/inference_on_a_image.py",
                "-c", config_path,
                "-p", checkpoint_path,
                "-i", image_path,
                "-t", text_prompts,
                "-o", os.path.join(output_dir, image_file)
            ]
            subprocess.run(command)



.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">
    Runs the trained model on validation images, saving predictions to the output directory.
    <p><span style="color:white;">'</p></span>



Visualize Results
---------------------------------------

.. code-block:: python

    selected_images = random.sample(os.listdir("/content/final_val_images"), 16)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for ax, image_file in zip(axes.flatten(), selected_images):
        img = Image.open(os.path.join("/content/final_val_images", image_file))
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()



.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>  
    - <strong>Objective</strong>: </span><span style="color:#000080;">
    Randomly selects and displays annotated images generated during inference for visual verification.
    <p><span style="color:white;">'</p></span>
    
.. figure:: /Documentation/images/output2.png
   :width:  700
   :align: center
   :alt: Alternative Text

.. raw:: html

    <p><span style="color:white;">'</p></span>
