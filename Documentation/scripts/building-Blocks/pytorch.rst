
.. figure:: /Documentation/images/Building-Blocks/log.png
   :width:  100
   :align: right
   :alt: Alternative Text

PyTorch
===========

.. raw:: html

   <p style="text-align: justify;">

   </p>


1. Introduction
------------------

.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">

  PyTorch is a powerful open-source machine learning library developed by Facebook's AI Research lab (FAIR). It provides a flexible and intuitive framework for building, training, and deploying deep learning models. PyTorch stands out for its dynamic computation graph mechanism, allowing for efficient gradient computation and enabling users to define and modify models on-the-fly.
  </span></p>

  <p style="text-align: justify;"><span style="color:#000080;">
  With PyTorch, developers can easily create various types of neural networks, including convolutional neural networks (CNNs), recurrent neural networks (RNNs), and transformers, among others. Its extensive collection of pre-built modules and utilities simplifies the process of building complex architectures for tasks such as image classification, object detection, natural language processing, and more.
  </span></p>

 <p style="text-align: justify;"><span style="color:#000080;">
  One of PyTorch's key strengths lies in its seamless integration with Python and NumPy, facilitating data manipulation and experimentation. Additionally, PyTorch provides support for GPU acceleration, enabling faster computation and training of deep learning models on compatible hardware.
  </span></p>

 <p style="text-align: justify;"><span style="color:#000080;">
  Whether you're a beginner exploring deep learning concepts or an experienced researcher developing cutting-edge models, PyTorch offers a rich ecosystem of tools, resources, and community support to accelerate your journey in the field of artificial intelligence.
 </span></p>



.. admonition::  Purpose

   .. container:: blue-box

    The purpose of this documentation is to provide a comprehensive introduction to tensors in PyTorch,emphasizing their importance and usage within the context of machine learning models.




2. Installing PyTorch with Anaconda
---------------------------------

.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
    Open the Anaconda prompt or terminal.
   </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">
    Create a new conda environment for PyTorch by running the following command:
   </span></p>



```
conda create --name pytorch_env
```


.. raw:: html
    
    <p style="text-align: justify;"><span style="color:#000080;">   
    This will create a new environment named <strong> pytorch_env </strong> .

    </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

    Activate the new environment by running the following command:
    </span></p>

```
conda activate pytorch_env
```


.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">

    Install PyTorch using conda. The following command installs the CPU version of PyTorch:
    </span></p>



```
conda install pytorch torchvision cpuonly -c pytorch
```


.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">

    If you have a GPU and want to install the GPU version of PyTorch, replace <strong>cpuonly</strong> with <strong>cudatoolkit</strong>. For example:
    </span></p>



```
conda install pytorch torchvision cudatoolkit -c pytorch
```



.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">

    This will install the necessary packages for PyTorch to run on your system.
    </span></p>

   <p style="text-align: justify;"><span style="color:#000080;">

    Verify that PyTorch is installed correctly by running the following command:
    
    </span></p>



```
python -c "import torch; print(torch.__version__)"
```




.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">

    This should print the version number of PyTorch that you just installed.
    </span></p>



3. Introduction to Tensors
---------------------------


.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
    Tensors are specialized data structures similar to arrays and matrices, used to encode the inputs, outputs, and
    parameters of a model in PyTorch. They are optimized for computation on GPUs and automatic differentiation.
    </span></p>


.. code-block:: python

    import torch

    # Create a tensor
    x = torch.tensor([[1, 2], [3, 4]])
    print(x)


* **Initializing Tensors**


.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
    Tensors can be initialized in various ways, including directly from data, from NumPy arrays, or from other tensors.
    Initializing tensors is flexible and intuitive, simplifying the process of tensor creation.
    </span></p>



.. code-block:: python

    import torch
    import numpy as np

    # Initialize from data
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)

    # Initialize from NumPy array
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)

    print(x_data)
    print(x_np)


* **Attributes of Tensors**


.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
    Tensor attributes include their shape, data type, and the device on which they are stored. These attributes are useful
    for understanding and manipulating tensors effectively.
    </span></p>


.. code-block:: python

    import torch

    # Create a tensor
    tensor = torch.rand(3, 4)

    # Get tensor attributes
    print(f"Shape of tensor: {tensor.shape}")
    print(f"Datatype of tensor: {tensor.dtype}")
    print(f"Device tensor is stored on: {tensor.device}")


* **Operations on Tensors**


.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
    PyTorch offers a wide range of tensor operations, including arithmetic operations, linear algebra, matrix manipulation,
    sampling, and more. Tensors can also be used for operations in GPU mode, providing optimized performance.
    </span></p>

.. code-block:: python

    import torch

    # Arithmetic operations
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.tensor([[5, 6], [7, 8]])

    # Matrix multiplication
    z1 = x @ y
    z2 = torch.matmul(x, y)

    print(z1)
    print(z2)

* **Bridge with NumPy**

.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
    Tensors in PyTorch can share their underlying memory with NumPy arrays, enabling seamless conversion between the two.
    This allows for smooth integration between PyTorch and NumPy, facilitating work with data.
    </span></p>

.. code-block:: python

    import torch
    import numpy as np

    # Tensor to NumPy array
    tensor = torch.tensor([1, 2, 3, 4])
    numpy_array = tensor.numpy()

    # NumPy array to Tensor
    numpy_array = np.array([5, 6, 7, 8])
    tensor = torch.from_numpy(numpy_array)

    print(tensor)

.. note::

    **For more practice and to learn more, we can visit this tutorial.**

    `Find the link to github repository <https://github.com/imadmlf/Learn_PyTorch_for_beginners./blob/main/lpytorch/tensors.ipynb>`__

    `Find the link to colab <https://colab.research.google.com/github/imadmlf/Learn_PyTorch_for_beginners./blob/main/lpytorch/tensors.ipynb>`__



4. Datasets & DataLoaders
---------------------------


.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
    PyTorch provides two important primitives for working with datasets: torch.utils.data.Dataset and torch.utils.data.DataLoader. These enable us to decouple dataset processing from model training code, enhancing readability and modularity.
    </p>

* Dataset:


.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
    Stores samples and their corresponding labels.
    Allows for custom transformations.
    Subclasses can be created for specific datasets.
    </span></p>

* DataLoader:


.. raw:: html


   <p style="text-align: justify;"><span style="color:#000080;">
    Wraps an iterable around the dataset.
    Facilitates easy access to samples during training.

    </span></p>

* **Loading a Dataset**


.. raw:: html


   <p style="text-align: justify;"><span style="color:#000080;">
    PyTorch also offers pre-loaded datasets, such as FashionMNIST, for prototyping and benchmarking models. These datasets subclass torch.utils.data.Dataset and implement specific functions for handling the data.
    For example, to load the Fashion-MNIST dataset using TorchVision:
    </span></p>


.. code-block:: python

    import torch
    from torch.utils.data import Dataset
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    import matplotlib.pyplot as plt


    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )



* **Iterating and Visualizing the Dataset**


.. raw:: html


  <p style="text-align: justify;"><span style="color:#000080;">
    We can index Datasets manually like a list: training_data[index]. We use matplotlib to visualize some samples in our training data.
    </span></p>


.. code-block:: python

    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
        }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

* output
This code generates a grid of images with their corresponding labels from the Fashion-MNIST dataset. Each image represents a clothing item, and the labels indicate the category of the clothing.

.. figure:: /Documentation/images/Building-Blocks/output.jpg
   :width: 400
   :align: center
   :alt: Alternative Text


* **Creating a Custom Dataset for Your Files**

.. raw:: html


  <p style="text-align: justify;"><span style="color:#000080;">

    To create a custom Dataset class, you must implement three functions: <span style="color:blue;">__init__</span>, <span style="color:blue;">__len__</span>, and <span style="color:blue;">__getitem__</span>. Below is an implementation example where the FashionMNIST images are stored in a directory (`img_dir`), and their labels are stored separately in a CSV file (`annotations_file`).
    </span></p>


.. code-block:: python

    import os
    import pandas as pd
    from torchvision.io import read_image
    from torch.utils.data import Dataset

    class CustomImageDataset(Dataset):
        def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
            self.img_labels = pd.read_csv(annotations_file)
            self.img_dir = img_dir
            self.transform = transform
            self.target_transform = target_transform

        def __len__(self):
            return len(self.img_labels)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
            image = read_image(img_path)
            label = self.img_labels.iloc[idx, 1]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label

__init__


.. raw:: html

  <p style="text-align: justify;"><span style="color:#000080;">
    

    The <span style="color:blue;">__init__</span> function is called once when instantiating the Dataset object. It initializes the directory containing the images, the annotations file, and both transforms.
    
    </span></p>
__len__


.. raw:: html

  <p style="text-align: justify;"><span style="color:#000080;">
    
    The <span style="color:blue;">__len__</span> function returns the number of samples in the dataset.
    
    </span></p>


Example:

.. code-block:: python

    def __len__(self):
        return len(self.img_labels)


__getitem__


.. raw:: html

  <p style="text-align: justify;"><span style="color:#000080;">
    
    The <span style="color:blue;">__getitem__</span> function loads and returns a sample from the dataset at the given index `idx`. It identifies the image’s location on disk based on the index, converts that to a tensor using `read_image`, retrieves the corresponding label from the CSV data, applies transform functions (if applicable), and returns the tensor image and corresponding label in a tuple.
    </span></p>


Example:

.. code-block:: python

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


* **Preparing Your Data for Training with DataLoaders**


.. raw:: html

  <p style="text-align: justify;"><span style="color:#000080;">
    The Dataset retrieves features and labels one sample at a time. When training a model, it's common to pass samples in minibatches, reshuffle the data at every epoch to reduce model overfitting, and use multiprocessing to speed up data retrieval.

    `DataLoader` is an iterable that abstracts this complexity for us in an easy API.
    </span></p>

.. code-block:: python

    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

* **Iterate Through the DataLoader**


.. raw:: html

  <p style="text-align: justify;"><span style="color:#000080;">

    After loading the dataset into the DataLoader, you can iterate through the dataset as needed. Each iteration returns a batch of `train_features` and `train_labels`. Since `shuffle=True`, the data is shuffled after iterating over all batches.
    </span></p>

Example:

.. code-block:: python

    # Display image and label.
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")

* output

This code segment outputs a batch of training features and their corresponding labels from the train_dataloader.

.. figure:: /Documentation/images/Building-Blocks/output1.jpg
   :width: 400
   :align: center
   :alt: Alternative Text

.. note::

    **For more practice and to learn more, we can visit this tutorial.** 

    `Find the link to Github repository <https://github.com/imadmlf/Learn_PyTorch_for_beginners./blob/main/lpytorch/data.ipynb>`__
    
    `Find the link to colab <https://colab.research.google.com/github/imadmlf/Learn_PyTorch_for_beginners./blob/main/lpytorch/data.ipynb>`__


 