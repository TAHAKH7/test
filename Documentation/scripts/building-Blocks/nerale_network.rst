Neural Network
===============


1. Introduction
------------------

In this article, we will build a neural network from scratch and use it to classify

.. raw:: html


  <p style="text-align: justify;"><span style="color:#000080;">
    A neural network is a type of machine learning algorithm that forms the foundation of various artificial intelligence applications such as computer vision, forecasting, and speech recognition. It consists of multiple layers of neurons, with each layer being activated based on inputs from the previous layer. These layers are interconnected by weights and biases, which determine how information flows through the network. While neural networks are often compared to biological neural networks found in the brain, it's important to exercise caution when making such comparisons, as artificial neural networks are simplified representations designed for specific computational tasks.
  </span></p>


.. figure:: /Documentation/images/Building-Blocks/neral.webp
   :width:  700
   :align: center
   :alt: Alternative Text


.. raw:: html


  <p style="text-align: justify;"><span style="color:#000080;">
    The first layer is the input layer. Input layer activations come from the input to the neural network. The final layer is the output layer. The activations in the output layer are the output of the neural network. The layers in between are called hidden layers.
  </span></p>



2. dataset:
---------

.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">

    To read data from a CSV file using pandas (pd), you can use the read_csv function:
   </span></p>


.. code-block:: python

    import pandas as pd 
    df = pd.read_csv('cancer_classification.csv')

.. note::
    
    You can view the dataset and access  by clicking the `link to the dataset <https://github.com/imadmlf/taskes/blob/main/cancer_classification.csv>`__

    Read the `dataset <https://github.com/imadmlf/taskes/blob/main/cancer_classification.csv>`__ :
   
   

3. Training on a simple dataset
-----------------------------

1. `DataPreprocessing <https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataPreprocessing.py>`__ : 

.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
    This module likely contains functions or classes for preparing your raw data for analysis. This can include tasks such as handling missing values, encoding categorical variables, scaling numerical features, and splitting the data into training and testing sets.
    
    </span></p>
.. note::

    You can view the code and access  by clicking the. `link to DataPreprocessing class <https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataPreprocessing.py>`__
    
2. `DataExploration <https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataExploration.py>`__ :

.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
    This part of your pipeline focuses on understanding the structure and characteristics of your dataset. It might include functions or classes for displaying basic statistics (like mean, median, standard deviation), visualizations (like histograms, scatter plots, or correlation matrices), and checking for any anomalies or inconsistencies in the data.
    
   </span></p>

.. note::

    You can view the code and access  by clicking the.
     `link to the DataExploration class <https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataExploration.py>`__


3. `ModelTraining <https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/modeltrainer.py>`__ : 

.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
    Here, you're training a machine learning model on your preprocessed data. This typically involves selecting an appropriate algorithm (like a neural network), defining a loss function, and optimizing model parameters using an optimization algorithm (like stochastic gradient descent).
    
   </span></p>

.. note::

    You can view the code and access  by clicking the  `link to the ModelTraining class <https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/modeltrainer.py>`__.

    
4. `ModelEvaluation <https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/ModelEvaluation.py>`__ :

 
 .. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
    After training your model, you need to evaluate its performance. This module likely contains functions or classes for computing various evaluation metrics (like accuracy, precision, recall, or F1-score), generating confusion matrices, and visualizing prediction results.
   
   </span></p>
.. note:: 

     You can view the code and access  by clicking the `link to the ModelEvaluation class  <https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/ModelEvaluation.py>`__


5. `NeuralNetwork <https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/neural_network.py>`__    :

.. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">
    This appears to be a class for defining a neural network architecture using the PyTorch library. It specifies the layers, activation functions, and connections between neurons in the network.
    
   </span></p>
.. note::

    'You can view the code and access by clicking the  `link to the NeuralNetwork class <https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/neural_network.py>`__.


.. code-block::python
    from DataPreprocessing import DataPreprocessing
    from DataExploration import DataExploration
    from ModelEvaluation import ModelEvaluation
    from ModelTraining import ModelTraining
    from neural_network import NeuralNetwork
    import torch



4. Test the `DataPreprocessing <https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataPreprocessing.py>`__  class
-------------------------------------------------------------------------------------------------------------------------


The `preprocessor` object is created using the `DataPreprocessing`_ class, which prepares the data for training a machine learning model. After splitting the data into training and testing sets using the `split_data()`_ method, it normalizes the data with `normalize_data()`_. Finally, it converts the data into tensors with `tensorize_data()`_, ready for model training and evaluation.

.. code-block:: python

    preprocessor = DataPreprocessing(df)
    x_train, x_test, y_train, y_test = preprocessor.split_data(test_size=0.2, random_state=42)
    x_train, x_test = preprocessor.normalize_data()
    x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor = preprocessor.tensorize_data()

.. _`DataPreprocessing`: https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataPreprocessing.py
.. _`split_data()`: https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataPreprocessing.py#LX
.. _`normalize_data()`: https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataPreprocessing.py#LX
.. _`tensorize_data()`: https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataPreprocessing.py#LX



5. test the `DataExploration <https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/DataExploration.py>`__ class:
------------------------------------------------------------------------------------------------------------------------


* **information_help()**: Their role is to display the methods existing in the DataExploration class.


.. code-block:: python

    explorer = DataExploration(df)
    explorer.information_help()




*output:*

 .. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">

    <span style="color:blue;">DisplayData()</span>:Display the first few rows of the DataFrame.
    </span></p>
    <p style="text-align: justify;"><span style="color:#000080;">

    <span style="color:blue;">DisplayDataTypes() </span>:Display the data types of each column in the DataFrame.
     </span></p>   
    <p style="text-align: justify;"><span style="color:#000080;">

    <span style="color:blue;">DisplayDataInfo() </span>:Display information about the DataFrame, including number of rows, columns, and data types.
     </span></p>   
    <p style="text-align: justify;"><span style="color:#000080;">
    <span style="color:blue;">DisplayDataDescription() </span>:Display descriptive statistics for each column of the DataFrame.

    </span></p>
    <p style="text-align: justify;"><span style="color:#000080;">
    <span style="color:blue;">DisplayCorrelationMatrix()</span> :Display the correlation matrix between all numeric columns of the DataFrame.
    </span></p>
    <p style="text-align: justify;"><span style="color:#000080;">

    <span style="color:blue;">DisplayCorrelationWithColumn(column)</span>:correletion with a specific column
    </span></p>
     <p style="text-align: justify;"><span style="color:#000080;"> 

    <span style="color:blue;">DisplayHeatMap() </span>:Displays a heatmap of the correlation matrix.
    </span></p>
    <p style="text-align: justify;"><span style="color:#000080;">

    <span style="color:blue;">DisplayPairPlot() </span>:This method creates a pairplot, also known as a scatterplot matrix, which shows pairwise relationships between numerical columns 
    </span></p>
    <p style="text-align: justify;"><span style="color:#000080;">

    <span style="color:blue;">DisplayCountPlot() </span>:This method generates a countplot, which is a type of bar plot that shows the frequency of each category in a categorical column of the DataFrame
     </span></p>   
    <p style="text-align: justify;"><span style="color:#000080;">

    <span style="color:blue;">DisplayBoxPlot()</span>:This method creates a boxplot for a numerical column in the DataFrame.

    <p style="text-align: justify;"><span style="color:#000080;">

    <span style="color:blue;">DisplayScatterPlot() </span>:This method generates a scatter plot between two numerical columns in the DataFrame
    </span></p>    
    <p style="text-align: justify;"><span style="color:#000080;">
    
    <span style="color:blue;">DisplayHistogram()</span>:This method creates a histogram for a numerical column in the DataFrame
    </span></p>
    

* **DisplayData()**: Displays the head of the DataFrame.


.. code-block:: python

    explorer = DataExploration(df)
    print("DataFrame Head")
    explorer.DisplayData()


* **DisplayDataTypes()**: Displays the data types of columns in the DataFrame.

.. code-block:: python

    print("\nData Types")
    explorer.DisplayDataTypes()


* **DisplayDataInfo()** : Displays general information about the DataFrame.

.. code-block:: python
    
    print("\nData Info")
    explorer.DisplayDataInfo()

* **DisplayDataDescription()** : Displays statistical descriptions of the data.

.. code-block:: python

    print("\nData Description")
    explorer.DisplayDataDescription()

* **DisplayDataShape()** :Displays the shape of the DataFrame.

.. code-block:: python

    print("\nData Shape")
    explorer.DisplayDataShape()


* **DisplayMissingValues()**:Displays information about missing values in the DataFrame.


.. code-block:: python

    print("\nMissing Values")
    explorer.DisplayMissingValues()    

* **DisplayCorrelationMatrix()** :Displays the correlation matrix of numerical features in the DataFrame.


.. code-block:: python

    print("\nCorrelation Matrix")
    explorer.DisplayCorrelationMatrix()

* **DisplayCorrelationWithColumn('benign_0__mal_1')** :Displays the correlation of all features with the target column named 'benign_0__mal_1'.

.. code-block:: python
    
    print("\nCorrelation with 'target' column:")
    explorer.DisplayCorrelationWithColumn('benign_0__mal_1')

* **DisplayHeatMap()** :Displays a heatmap of the correlation matrix.


.. code-block:: python

    print("\nHeatMap")
    explorer.DisplayHeatMap()





5. test `the NeuralNetwork <https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/neural_network.py>`__  class
-------------------------------------------------------------------------------------------------------------



.. code-block:: python

    input_features = len(df.columns) - 1
    out_features = df['benign_0__mal_1'].unique().sum()
    neural_net = NeuralNetwork(input_features, out_features)
    print("Neural Network Architecture:")
    print(neural_net)
 


`output`:


Neural Network Architecture:


.. figure:: /Documentation/images/Building-Blocks/neuralnetwork_output.jpg
   :width: 100%
   :align: center
   :alt: Alternative text for the image
   :name: Architecture


Here's the explanation:
 .. raw:: html

   <p style="text-align: justify;"><span style="color:#000080;">

    <span style="color:blue;">input_features = len(df.columns) - 1</span>: This line calculates the number of input features for the neural network. It subtracts 1 from the total number of columns in the DataFrame `df` to exclude the target column (assuming the target column is named `'benign_0__mal_1'`).
    </span></p>   
   <p style="text-align: justify;"><span style="color:#000080;">

    <span style="color:blue;">out_features = df['benign_0__mal_1'].unique().sum()</span>: This line calculates the number of output features for the neural network. It first extracts the unique values from the target column `'benign_0__mal_1'` using the `unique()` method. Then, it sums up these unique values, which would typically represent the number of classes or categories in a classification task.
    </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

    <span style="color:blue;">neural_net = NeuralNetwork(input_features, out_features)</span>: This line creates an instance of the `NeuralNetwork` class with the calculated number of input and output features.
    </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

    <span style="color:blue;">print("Neural Network Architecture") </span>: This line simply prints a message indicating that the following print statement will display the architecture of the neural network.
    </span></p>
   <p style="text-align: justify;"><span style="color:#000080;">

    <span style="color:blue;">print(neural_net)</span>: This line prints the architecture of the neural network instance `neural_net`. The architecture of the neural network is typically defined by the layers and their configurations, which are specified within the `NeuralNetwork` class. Therefore, printing `neural_net` will display its architecture, including the layers, activation functions, and other configurations specified during its initialization.
    </span></p>

6. Testing the  `ModelTraining <https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/modeltrainer.py>`__  class
--------------------------------------------------------------------------------------------------------------------

This code snippet demonstrates setting up the neural network model, defining the loss function and optimizer, and then training the model using a ModelTrainer class. During training, it collects the training and testing losses for each epoch.



.. code-block:: python

    from torch import nn
    model = neural_net
    criterion = nn.BCELoss()   
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 
    from modeltrainer import ModelTrainer
    trainer = ModelTrainer(model, criterion, optimizer)
    train_losses, test_losses = trainer.train(x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor, epochs=600)



plot train_losses and test_losses


.. code-block:: python
    trainer.plot_loss(train_losses, test_losses)



.. figure:: /Documentation/images/Building-Blocks/training.jpg
   :width: 100%
   :alt: Alternative text for the image
   :name: logo




7. test the `ModelEvaluation <https://github.com/imadmlf/Neural_Network_Wrapper/blob/main/ModelEvaluation.py>`__  class 
------------------------------------------------------------------------------------------------------------------------


.. code-block:: python

    evaluator = ModelEvaluation(model, criterion, optimizer)


.. code-block:: python

        model.eval()
        with torch.no_grad():
            y_pred = model(x_test_tensor)
            y_pred = (y_pred > 0.5).float()    



.. code-block:: python

    evaluator.confusion_matrix(y_test_tensor, y_pred)



.. figure:: /Documentation/images/Building-Blocks/conf.jpg
   :width: 100%
   :alt: Alternative text for the image
   :name: logo



Link  to github repository and colab applications:
-----------------------------------------------------

.. note::
    
    **For more practice and to learn more, we can visit this tutorial.**

    `Find the link to github repository <https://github.com/imadmlf/Neural_Network_Wrapper>`__


    `Link to Colab notebook <https://colab.research.google.com/github/imadmlf/Learn_PyTorch_for_beginners./blob/main/test.ipynb>`__


    `Link to Colab notebook  <https://colab.research.google.com/github/imadmlf/Learn_PyTorch_for_beginners./blob/main/NereulNe.ipynb>`__ 


