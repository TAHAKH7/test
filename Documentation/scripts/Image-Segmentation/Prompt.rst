Prompt Generator / Analyzer 
=============================

------------------------------------------------------------------------------



.. admonition::  contents 

   .. container:: blue-box

    * `Overview <https://images-segmmentation-prompt.readthedocs.io/fr/latest/Documentation/scripts/Image-Segmentation/Prompt.html#overview>`__
    
    * `Prompt Gemini Generator <https://images-segmmentation-prompt.readthedocs.io/fr/latest/Documentation/scripts/Image-Segmentation/Prompt.html#prompt-gemini-generator>`__

    * `Features <https://images-segmmentation-prompt.readthedocs.io/fr/latest/Documentation/scripts/Image-Segmentation/Prompt.html#features>`__

        - `Prompt pre-processing <https://images-segmmentation-prompt.readthedocs.io/fr/latest/Documentation/scripts/Image-Segmentation/Prompt.html#prompt-pre-processing>`__

        - `Similarity Reduction <https://images-segmmentation-prompt.readthedocs.io/fr/latest/Documentation/scripts/Image-Segmentation/Prompt.html#similarity-reduction>`__

        - `Complexity Analysis <https://images-segmmentation-prompt.readthedocs.io/fr/latest/Documentation/scripts/Image-Segmentation/Prompt.html#complexity-analysis>`__

        - `Readability Analysis <https://images-segmmentation-prompt.readthedocs.io/fr/latest/Documentation/scripts/Image-Segmentation/Prompt.html#readability-analysis>`__

    * `Prompt Generator Example <https://images-segmmentation-prompt.readthedocs.io/fr/latest/Documentation/scripts/Image-Segmentation/Prompt.html#prompt-generator-example>`__

    * `Prompt Analyzer Exemple <https://images-segmmentation-prompt.readthedocs.io/fr/latest/Documentation/scripts/Image-Segmentation/Prompt.html#prompt-analyzer-exemple>`__

.. raw:: html

    <p><span style="color:white;">'</p></span>


.. figure:: /Documentation/images/IMAGE/promptAnalyzer.jpg
   :width: 100
   :align: center
   :alt: Alternative text for the image


Overview
----------


.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>

    The prompt_analyzer</span><span style="color:#000080;"> is designed to analyze sets of prompts associated with images and generated using gemini pro vision model. After properly </span><span style="color:blue;">processing the prompts</span><span style="color:#000080;">, removing similarities based on user set threshold, </span><span style="color:blue;">the prompt_analyzer</span><span style="color:#000080;"> evaluates them based on complexity and readability to identify the most effective prompts.
    It leverages various Python libraries including NLTK for natural language processing, scikit-learn for feature extraction and cosine similarity, and others for specific linguistic tasks.
   </i></span></p>

    <p><span style="color:white;">'</p></span>
    
Prompt Gemini Generator
-------------------------

.. raw:: html

    <p style="text-align: justify;"><span style="color:#000080;"><i>
    The prompt_generator class is designed to automate the creation of textual prompts for images using </span><span style="color:blue;">Gemini pro Vision API.

    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    
    </i></span></p>
    The prompt_generator class automates the process of generating text from images, offering a bridge between visual content and textual descriptions through advanced machine learning techniques.
    <p style="text-align: justify;"><span style="color:#000080;"><i>

**Parameters:**

.. raw:: html

    </i></span></p>
    <p style="text-align: justify;"><span style="color:blue;"><i>
    - Model:</span><span style="color:#000080;"> Identifier or configuration for a generative model.

    </i></span></p>
    <p style="text-align: justify;"><span style="color:blue;"><i>
    - key:</span><span style="color:#000080;"> API key for accessing the generative model's service.

    </i></span></p>
    <p style="text-align: justify;"><span style="color:blue;"><i>
    - Images_dir:</span><span style="color:#000080;"> Directory path where image files are stored.

    </i></span></p>
    <p style="text-align: justify;"><span style="color:blue;"><i>
    - Images_extensions:</span><span style="color:#000080;"> List of image file extensions to consider.
    </i></span></p>

    <p><span style="color:white;">'</p></span>

**Operations:**

.. raw:: html

    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    . Configures the generative model with the provided API key (gai.configure).
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    . Configures the generative model with the provided API key (gai.configure).
    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    . Retrieves and stores paths to images within the specified directory that match the given extensions using sv.list_files_with_extensions.
      </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    .  Initializes a dictionary (prompts_dict) to store the generated prompts indexed by image name.
      </i></span></p>

    <p><span style="color:white;">'</p></span>

**Prompt Generation (generate_prompts):**

.. raw:: html

    <span style="color:blue;"><strong>Parameter:</strong>
    </span>
    <p style="text-align: justify;"><span style="color:blue;"><i>
    . number_of_prompts:</span><span style="color:#000080;"> Specifies how many prompts to generate per image.
     </i></span></p>

    <span style="color:blue;"><strong>Operations:</strong></span>

    <p style="text-align: justify;"><span style="color:#000080;"><i>
    . Iterates over each image file retrieved during initialization.
     </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    . For each image, it opens the image file and generates the specified number of prompts using the configured model.
     </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    . Each prompt's text is added to prompts_dict under the corresponding image name.
     </i></span></p>

    <span style="color:blue;"><strong>Output:</strong></span>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    . Returns prompts_dict, a dictionary where each key is an image name and the value is a list of generated prompts for that image.
     </i></span></p>
    <span style="color:blue;"><strong>Key Functionalities:</strong></span>


    <p style="text-align: justify;"><i>
    . <span style="color:blue;">Image Handling:</span> <span style="color:#000080;">Opens image files and prepares them for prompt generation.
     </i></span></p>
    <p style="text-align: justify;"><i>
    . <span style="color:blue;">Prompt Generation:</span><span style="color:#000080;">Leverages a deep learning model to generate creative or descriptive text based on the image content.
     </i></span></p>
    <p style="text-align: justify;"><i>
    . <span style="color:blue;">Data Management: </span><span style="color:#000080;">Efficiently manages and catalogs prompts for multiple images, facilitating easy retrieval and usage.
     </i></span></p>

Features
----------

.. raw:: html

    <p style="text-align: justify;"><i>
    . <span style="color:blue;">Prompt processing : </span><span style="color:#000080;">Removes stop words and puntuation to help ensure similarity comparison.
     </i></span></p>
     
    <p style="text-align: justify;"><i>
    . <span style="color:blue;">Similarity Reduction: </span><span style="color:#000080;">Removes highly similar prompts to ensure diversity using cosine similarity.


     </i></span></p>
    

    <p style="text-align: justify;"><i>
    . <span style="color:blue;">Complexity Analysis:  </span><span style="color:#000080;">
    Evaluates the complexity of prompts based on the length and vocabulary richness.
         </i></span></p>


    <p style="text-align: justify;"><i>
    . <span style="color:blue;">Readability Analysis: </span><span style="color:#000080;">
     Computes readability scores using the Flesch Reading Ease formula.
         </i></span></p>


Prompt pre-processing
______________________

.. code-block:: python

    def prompt_processing(self)

.. raw:: html

    </i></span></p>
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    The prompt_processing performs preprocessing on a list of text prompts to prepare them for further analysis.
        </i></span></p>


    </i></span></p>

* **Core Functionality:**


.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Remove Punctuation:</strong></span><span style="color:#000080;">Each prompt is stripped of punctuation using a translation table, which simplifies the text and removes unnecessary characters.
        </i></span></p>
    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Tokenization:</strong></span><span style="color:#000080;">The unpunctuated prompt is then split into individual words (tokens) using NLTK’s word_tokenize.
        </i></span></p>
    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Remove Stop Words:</strong></span><span style="color:#000080;"> Common words (like "and", "the", etc.) that do not add much value in text analysis (known as stop words) are filtered out from the tokens.
        </i></span></p>
    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Track Lengths and Unique Words: </strong></span><span style="color:#000080;">The method calculates the length of each filtered prompt (number of meaningful words) and identifies the unique words used in each prompt.
        </i></span></p>


* **Output**

.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Prompts_unpunctuated:</strong></span><span style="color:#000080;"> List of prompts with punctuation removed.
        </i></span></p>
    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Prompts_length: </strong></span><span style="color:#000080;">List of prompts after removing stop words.
        </i></span></p>
    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Unique_words_list:</strong></span><span style="color:#000080;">List containing the length of each filtered prompt.
        </i></span></p>
    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>unique_words_list: </strong></span><span style="color:#000080;">List of sets, each containing unique words from each prompt.
        </i></span></p>

Similarity Reduction
______________________


.. code-block:: python

    def prompts_similarity(self, remove_similar=False, threshold=0.7):




.. raw:: html


    <p style="text-align: justify;"><span style="color:#000080;"><i>
    The prompts_similarity method evaluates the similarity between text prompts and optionally removes highly similar ones based on a specified threshold (set by default as 70% similarity, meaning that for 10 prompts with similarity rate higher than 70%, only one will remain).
        </i></span></p>

* **Functionality:**

.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Preprocessing:</strong></span><span style="color:#000080;">It first processes the list of prompts to remove punctuation, using the prompt_processing method.
        </i></span></p>

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Vectorization:</strong></span><span style="color:#000080;">
    Converts the cleaned prompts into a TF-IDF matrix, which numerically represents the importance of words within the prompts.
        </i></span></p>

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Similarity Calculation:</strong></span><span style="color:#000080;">Computes pairwise cosine similarities between all prompts, resulting in a similarity matrix.
        </i></span></p>


* **Parameters:**

.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>remove_similar (boolean): </strong></span><span style="color:#000080;">If set to True, the method will remove prompts that are similar above a certain threshold.
        </i></span></p>

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>threshold (float): </strong></span><span style="color:#000080;">The similarity threshold for determining whether two prompts are considered similar.
        </i></span></p>


* **Output**

.. raw:: html
    
    <p style="text-align: justify;"><span style="color:#000080;"><i>
        - If remove_similar is False, the method returns the similarity matrix.
    
        </i></span></p>

    <p style="text-align: justify;"><span style="color:#000080;"><i>
        If remove_similar is True, it modifies the list of prompts by removing similar ones: Identifies pairs of prompts that exceed the similarity threshold. Removes prompts to reduce redundancy, keeping one prompt from each similar pair, and returns the pruned list of prompts.
    </i></span></p>

* **Use Case:**

.. raw:: html
    
    <p style="text-align: justify;"><span style="color:#000080;"><i>
     This method is useful for reducing redundancy in datasets where prompts may be too similar, which can be essential for training models where diversity of input data enhances learning efficacy.

        </i></span></p>

Complexity Analysis
________________________

.. code-block:: python

    def prompt_complexity(self):

.. raw:: html
    
    <p style="text-align: justify;"><span style="color:#000080;"><i>
     The prompt_complexity method calculates the complexity of text prompts based on their length and vocabulary richness.
        </i></span></p>

* **Functionality:**

.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Preprocessing: </strong></span><span style="color:#000080;">It starts by calling prompt_processing to get a list of prompts that have been filtered of punctuation and stop words.
        </i></span></p>

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Complexity Calculation:</strong></span><span style="color:#000080;">
        </i></span></p>

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Prompt Length: </strong></span><span style="color:#000080;">Measures the number of words in each filtered prompt.
        </i></span></p>

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Unique Words:</strong></span><span style="color:#000080;">Counts the number of unique words in each prompt.
        </i></span></p>

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Vocabulary Richness: </strong></span><span style="color:#000080;"> Calculates the ratio of unique words to total prompt length, which indicates the diversity of vocabulary used.
        </i></span></p>

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Complexity Score:</strong></span><span style="color:#000080;">Multiplies the prompt length by the vocabulary richness to get a score representing the prompt's complexity.
        </i></span></p>

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Compilation of Scores:</strong></span><span style="color:#000080;">Stores and then returns a list of these complexity scores for each prompt, sorted from least to most complex.
        </i></span></p>

* **Output**


.. raw:: html
    
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    Returns a sorted list of complexity scores, where each score quantifies the lexical richness and length of a prompt, serving as an indicator of its complexity.

        </i></span></p>

* **Use Case:**

.. raw:: html
    
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    This method is valuable for analyzing and ranking prompts based on their linguistic complexity, which can be important for applications where the level of language complexity is critical, such as educational content creation or text-based AI training scenarios.
        </i></span></p>


Readability Analysis
_____________________

.. code-block:: python

    def readability(self):



.. raw:: html
    
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    
    The readability method calculates the readability of a text prompt using the Flesch Reading Ease formula, a widely recognized method to evaluate the ease of understanding of a text.

    </i></span></p>

* **Functionality**

.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Sentence and Word Tokenization: </strong></span><span style="color:#000080;">The method first tokenizes the input prompt into sentences and words using NLTK's sent_tokenize and word_tokenize.
        </i></span></p>


    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong> Syllable Counting: </strong></span><span style="color:#000080;">Retrieves the CMU Pronouncing Dictionary (cmudict) to count syllables. For each word, it extracts the pronunciation and counts the number of syllable markers (digits in the pronunciation).
        </i></span></p>
        
    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Flesch Score Calculation: </strong></span><span style="color:#000080;"> Computes the total number of sentences, words, and syllables in the prompt. Applies the Flesch Reading Ease formula 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words). Rounds the result to two decimal places.
        </i></span></p>


* **Output**


.. raw:: html
    
    <p style="text-align: justify;"><span style="color:#000080;"><i>

    Returns the Flesch Reading Ease score for the prompt, where higher scores indicate texts that are easier to read, and lower scores indicate texts that are more difficult.
    </i></span></p>


* **Use Case**

.. raw:: html
    
    <p style="text-align: justify;"><span style="color:#000080;"><i>

    This method is particularly useful for ensuring text content is appropriate for the intended audience's reading level, such as in educational materials, marketing content, or publishing, where readability can impact engagement and comprehension.

    </i></span></p>


Prompt processing
-------------------


.. raw:: html
    
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    The process_prompts method in performs several operations to analyze and rank text prompts based on either their complexity or readability (user's choice), and then records the top results. Here's a brief overview:
    </i></span></p>

* **Functionality**

.. raw:: html

    
    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Initialization:  </strong></span><span style="color:#000080;">It initializes an empty list to store results.
        </i></span></p>

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Loop Through Prompts:  </strong></span><span style="color:#000080;">Iterates through each set of prompts associated with an image
        </i></span></p>

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Similarity Removal:</strong></span><span style="color:#000080;">First removes similar prompts to ensure diversity using the prompts_similarity method. Scoring and Sorting
        </i></span></p>

    <p style="text-align: justify;"><i>
    - <span style="color:#000080;">If complexity is True, it calculates complexity scores using prompt_complexity and sorts prompts from most to least complex.
        </i></span></p>

    <p style="text-align: justify;"><i>
    - <span style="color:#000080;">If readability is True, it calculates readability scores using prompt_readability and sorts prompts from easiest to hardest to read.
        </i></span></p>

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong> Selection of Top Prompts:</strong></span><span style="color:#000080;">Extracts the top three prompts. If there are fewer than three, fills the remaining slots with "N/A".
        </i></span></p>

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Recording Results:</strong></span><span style="color:#000080;">Compiles results into a dictionary for each image, including the image name and the top three prompts.
        </i></span></p>

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Output to CSV:</strong></span><span style="color:#000080;">Calls write_to_csv to write the results to a CSV file.
        </i></span></p>


* **Parameter:**

.. raw:: html

    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>readability (bool):</strong></span><span style="color:#000080;">Determines if prompts should be analyzed and sorted by readability.
        </i></span></p>
    <p style="text-align: justify;"><span style="color:blue;"><i>
    - <strong>Complexity (bool):</strong></span><span style="color:#000080;">Determines if prompts should be analyzed and sorted by complexity.
        </i></span></p> 



* **Output:**

.. raw:: html
    
    <p style="text-align: justify;"><span style="color:#000080;"><i>

    Writes a CSV file named prompt_results.csv with headers image_name, best_prompt1, best_prompt2, and best_prompt3, documenting the top three prompts for each image.

    </i></span></p>


* **Use Case**

.. raw:: html
    
    <p style="text-align: justify;"><span style="color:#000080;"><i>
        This method is useful for preparing prompt datasets where the best prompts need to be identified and cataloged based on specific criteria like readability or complexity. It's particularly valuable in scenarios where prompt quality impacts user engagement or educational outcomes, ensuring that the most suitable prompts are used for further applications or studies.
    </i></span></p>


Prompt Generator Example
------------------------
.. raw:: html
    
    <p style="text-align: justify;"><span style="color:#000080;"><i>
        Here is a simple example to demonstrate how to use the prompt_generator class:
    </i></span></p>


.. code-block:: python

    API_Key=input("Enter your API Key")
    prompts=prompt_generator('gemini-pro-vision',API_Key)
    prompts_dict=prompts.generate_prompts(number_of_prompts=10)


Prompt Analyzer Exemple
------------------------

.. raw:: html
    
    <p style="text-align: justify;"><span style="color:#000080;"><i>
    Here is a simple example to demonstrate how to use the prompt_analyzer class:
    </i></span></p>


.. code-block:: python

    prompts_dict = {
        'image1.jpg': ['An early morning', 'Sunrise at the beach', 'Dawn breaks over the ocean']
    }
    analyzer = prompt_analyzer(prompts_dict)
    analyzer.process_prompts(complexity=True)

