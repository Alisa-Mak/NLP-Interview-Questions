# NLP-Interview-Questions

1. **Describe the types of tasks that are best suited for BERT, GPT, and BART/T5 based on their respective architectures?**

   BERT (Encoder-only)
      - **Architecture**: BERT (Bidirectional Encoder Representations from Transformers) is designed with only an encoder. It processes the input text bidirectionally, meaning it looks at the entire context of a word both to the left and the right, making it powerful for understanding contextual relationships.
   
      - **Best for**:
        - **Text Classification**: Sentiment analysis, spam detection, etc.
        - **Named Entity Recognition (NER)**: Identifying entities such as names, places, or organizations in text.
        - **Question Answering (QA)**: Extracting answers from a passage of text, e.g., SQuAD tasks.
        - **Part-of-Speech Tagging (POS)**: Assigning grammatical categories (noun, verb, etc.) to words in a sentence.
        - **Textual Similarity**: Determining how similar two pieces of text are (e.g., paraphrase detection).
        - **Embedding Generation**: Generating embeddings for tasks like search or clustering.
   
      - **Why?**: BERT’s encoder-only architecture is excellent for understanding and representing text at a deep level because it has a bidirectional view of the context around each word, making it strong for tasks requiring deep understanding of input text.
   
   GPT (Decoder-only)
      - **Architecture**: GPT (Generative Pre-trained Transformer) is based on a decoder-only architecture. It’s unidirectional, meaning it generates text by predicting the next word based on previous ones (left-to-right sequence generation). GPT is primarily used for language generation tasks.
   
      - **Best for**:
        - **Text Generation**: Writing essays, articles, or creative content.
        - **Storytelling and Dialogue Generation**: Conversational AI and chatbots.
        - **Autocompletion**: Completing sentences, emails, code, etc.
        - **Summarization (generative)**: Generating summaries of documents.
        - **Machine Translation (generative)**: Translating text from one language to another.
        - **Language Modeling**: Predicting the next word in a sequence (e.g., predictive text).
   
      - **Why?**: GPT’s decoder-only architecture is highly effective for generation tasks since it predicts one token at a time in a sequence. It’s designed to generate coherent and contextually relevant outputs by building on prior tokens, making it powerful for creative or sequential tasks.
   
   BART / T5 (Full Encoder-Decoder)
      - **Architecture**: BART and T5 are full encoder-decoder models, which makes them highly flexible and versatile for both understanding and generating text. The encoder processes the input, and the decoder generates output, making them ideal for sequence-to-sequence tasks.
   
      - **Best for**:
        - **Text Summarization**: Generating concise summaries from longer texts (both extractive and abstractive summarization).
        - **Machine Translation**: Translating text from one language to another.
        - **Text Generation**: Similar to GPT but with more flexibility due to the encoder-decoder architecture.
        - **Question Answering**: Similar to BERT but with the ability to generate more complex responses.
        - **Text Completion and Transformation**: Tasks like paraphrasing or text style transfer.
        - **Data-to-Text Generation**: Converting structured data (like tables or databases) into natural language text.
   
      - **Why?**: The encoder-decoder architecture of models like BART and T5 excels at both understanding and generating sequences. This makes them ideal for complex tasks where both input comprehension and output generation are required, such as summarization, translation, and text-to-text transformations.

2. **What is NLP, and why is it important?**  
   Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It's important because it enables machines to understand, interpret, and respond to human language, making applications like chatbots, translation, and voice assistants possible.

3. **Explain the difference between NLP and NLU (Natural Language Understanding).**  
   NLP is a broader field that deals with processing and analyzing human language, while NLU is a subset of NLP focused on understanding the meaning and context of language.

   **Natural Language Processing (NLP):**  
   NLP encompasses a wide range of tasks involving the processing and analysis of human language. It includes techniques and applications for understanding, interpreting, and generating text. 
   
   *Examples:*
   - **Machine Translation:** Translating text from one language to another, such as Google Translate.
   - **Named Entity Recognition (NER):** Identifying and categorizing entities like people, organizations, and locations in a text.
   - **Sentiment Analysis:** Determining the sentiment behind a piece of text, such as classifying a movie review as positive or negative.
   
   **Natural Language Understanding (NLU):**  
   NLU is a specific subset of NLP that focuses on understanding the meaning and context of language. It involves interpreting and extracting information from text, often to facilitate deeper interactions with users.
   
   *Examples:*
   - **Intent Recognition:** In a chatbot, determining if a user’s query is about booking a flight, checking weather, or seeking customer support.
   - **Slot Filling:** Extracting specific information from a user’s input, such as dates, locations, or names, to complete a booking request.
   - **Word Sense Disambiguation (WSD):** Identifying the correct meaning of a word in context, such as distinguishing between "bank" as a financial institution or the side of a river.

5. **What are some common applications of NLP?**  
   Common applications include machine translation, sentiment analysis, speech recognition, chatbots, text summarization, and document classification.

6. **Describe tokenization in NLP.**  
   Tokenization is the process of splitting text into smaller units, such as words or phrases (tokens), which are then analyzed in NLP tasks.

7. **What is stemming, and how does it differ from lemmatization?**  
   Stemming reduces words to their root forms by stripping suffixes, often resulting in non-standard words. Lemmatization, on the other hand, reduces words to their base form (lemma) based on dictionary definitions.

8. **Explain the concept of stop words in NLP.**  
   Stop words are common words (like "the," "is," "in") that are usually filtered out in NLP tasks because they provide little meaningful information.

9. **What is POS tagging, and why is it used?**  
   Part-of-Speech (POS) tagging assigns parts of speech (like noun, verb, adjective) to words in a sentence, helping understand sentence structure and meaning.

10. **How does named entity recognition (NER) work?**  
   NER identifies and classifies entities in text, such as people, organizations, locations, and dates. It uses statistical models and linguistic rules to label entities in context.

      **Preprocessing:** Text is cleaned and tokenized into smaller units like words or phrases.
      
      **Feature Extraction:** Various features (e.g., words, parts of speech, context) are extracted to help the model identify entities.
      
      **Model Application:** Statistical models or machine learning algorithms (like CRF or LSTM) are used to analyze these features. These models are often trained on annotated datasets where entities are labeled.
      
      **Entity Classification:** The model labels tokens or phrases as belonging to categories such as PERSON, ORGANIZATION, LOCATION, or DATE based on the patterns it has learned.
      
      *Example:*  
      In the sentence "Apple Inc. was founded by Steve Jobs in Cupertino," NER identifies "Apple Inc." as an ORGANIZATION, "Steve Jobs" as a PERSON, and "Cupertino" as a LOCATION.

11. **What is TF-IDF, and what is its significance in NLP?**  
   TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents. It increases with the word's frequency in a document but decreases with its frequency across multiple documents, helping identify key terms that are more unique to specific documents.

12. **Explain the concept of word embeddings.**  
   Word embeddings are dense vector representations of words where similar words have similar representations in a continuous vector space. They capture semantic relationships between words, allowing algorithms to understand meanings and similarities.

13. **What are some popular word embedding techniques?**  
   Popular word embedding techniques include Word2Vec, GloVe, and FastText. These methods transform words into vectors that preserve their contextual relationships.

14. **What is Word2Vec, and how does it work?**  
   Word2Vec is a neural network-based model that learns word embeddings by predicting word contexts. It operates using two main architectures: CBOW (Continuous Bag of Words) and Skip-gram, which train the model to predict either the target word from its context or the context from the target word.

15. **Describe the difference between CBOW and Skip-gram models in Word2Vec.**  
   CBOW predicts a word based on its surrounding context, while Skip-gram does the reverse by predicting the context words given a target word. CBOW is generally faster, while Skip-gram is better for capturing rare word relationships.

16. **What is GloVe (Global Vectors for Word Representation)?**  
   GloVe is a word embedding technique that combines matrix factorization and local context-based methods to produce word vectors. It captures both global and local word co-occurrence information to better represent the meaning and relationships of words.

17. **Explain the concept of language modeling.**  
   Language modeling is the process of predicting the probability of a sequence of words. It is used to determine the likelihood of sentences, improve language understanding, and assist in applications like speech recognition and machine translation.

18. **What is perplexity in language modeling?**  
   Perplexity is a metric used to evaluate the quality of a language model. It measures how well the model predicts a sample of text, with lower perplexity indicating better predictive performance.

19. **How does a recurrent neural network (RNN) differ from a feedforward neural network?**  
   RNNs have connections that form directed cycles, allowing them to maintain memory of previous inputs, making them suitable for sequence data. Feedforward networks have no cycles and process input in a single direction without retaining any past information.

   RNNs (Recurrent Neural Networks) are older approaches that were used to handle tasks in NLP before the advent of Transformers and large language models (LLMs)

21. **What are some limitations of traditional RNNs?**  
    Traditional RNNs struggle with long-range dependencies due to their inability to retain information over long sequences. They are also prone to the vanishing gradient problem, making training difficult for long sequences.

22. **What is the vanishing gradient problem in RNNs?**  
    The vanishing gradient problem occurs when gradients become very small during backpropagation, preventing the model from learning long-term dependencies effectively. This is common in deep networks and RNNs processing long sequences.

23. **Describe the structure and purpose of Long Short-Term Memory (LSTM) networks.**  
    LSTMs are a type of RNN designed to overcome the vanishing gradient problem. They have a memory cell that retains information over long time steps, with gates (input, output, and forget) controlling the flow of information. This structure allows LSTMs to capture long-term dependencies in sequences better than traditional RNNs.

24. **What is attention mechanism in NLP?**  
   The attention mechanism allows a model to focus on specific parts of an input sequence when making predictions, giving more weight to important words. It helps models handle long-range dependencies by selectively attending to relevant information, improving performance in tasks like translation and summarization.

25. **Explain the transformer architecture.**  
   The transformer architecture is a deep learning model designed for sequence processing tasks without relying on recurrence. It uses self-attention to process input sequences in parallel, making it more efficient than RNNs. Transformers consist of stacked encoder-decoder layers where the encoder maps inputs into embeddings, and the decoder generates the output sequence.

26. **What are the advantages of transformers over RNNs and LSTMs?**  
   Transformers have several advantages over RNNs and LSTMs, including:  
   - **Parallel processing**: Transformers process all tokens in a sequence simultaneously, whereas RNNs and LSTMs process sequentially, making transformers faster.
   - **Better handling of long-range dependencies**: The self-attention mechanism allows transformers to capture dependencies across long sequences more effectively than RNNs, which struggle with long-term memory.
   - **Scalability**: Transformers scale well to large datasets due to their parallelism.

25. **Describe the encoder-decoder architecture in sequence-to-sequence models.**  
   In sequence-to-sequence (seq2seq) models, the encoder reads an input sequence and encodes it into a fixed-length context vector, which represents the entire input. The decoder then generates an output sequence from this vector, one token at a time. This architecture is commonly used in machine translation, summarization, and text generation.
   Transformer-based Encoder-Decoder:
Transformers have become the dominant architecture, particularly after the introduction of the Transformer model. Unlike RNNs, transformers rely on self-attention mechanisms that process all elements of a sequence in parallel, enabling faster training and capturing longer-range dependencies more effectively. The Transformer Encoder-Decoder architecture is widely used in models like BERT (encoder-only), GPT (decoder-only), and BART/T5 (full encoder-decoder).

   * Encoder: The encoder consists of multiple layers of self-attention and feed-forward networks. Each layer processes the input in parallel, making the transformer highly efficient.
   
   * Decoder: The decoder also consists of self-attention layers, but it adds an additional layer to attend to the encoder’s output. This allows it to generate sequences while keeping track of the input sequence context.

26. **What is beam search in the context of sequence generation?**  
   Beam search is a heuristic search algorithm used in sequence generation tasks to explore multiple possible output sequences simultaneously. Instead of selecting only the best next token, beam search keeps track of several potential candidates (beams) and continues to expand the most promising sequences based on cumulative probability. It balances exploration and exploitation to improve output quality.

27. **Explain the concept of machine translation and some popular methods for it.**  
   Machine translation (MT) involves automatically translating text from one language to another using algorithms. Popular methods include:  
   - **Statistical Machine Translation (SMT)**: Uses statistical models based on large bilingual corpora to learn translation patterns.
   - **Neural Machine Translation (NMT)**: Employs deep learning models, such as seq2seq with attention, to translate text end-to-end.
   - **Transformers**: Modern NMT models, like Google's Transformer-based BERT or OpenAI's GPT, use attention mechanisms for improved translation performance.

28. **How does sentiment analysis work?**  
   Sentiment analysis classifies text based on the emotions or opinions expressed, such as positive, negative, or neutral. It typically involves pre-processing text, extracting features (like word embeddings or n-grams), and applying machine learning or deep learning models (e.g., logistic regression, SVM, or neural networks) to classify sentiment.

29. **What are some techniques for feature extraction in sentiment analysis?**  
   Techniques for feature extraction in sentiment analysis include:  
   - **Bag of Words (BoW)**: Represents text as a vector of word counts or binary presence/absence.
   - **TF-IDF**: Measures the importance of words in a document relative to the entire corpus.
   - **Word Embeddings**: Dense vector representations like Word2Vec or GloVe that capture semantic meaning.
   - **Part-of-Speech (POS) Tagging**: Identifies the grammatical role of words, which can help in understanding sentiment.

30. **What is topic modeling, and how is it useful in NLP?**  
   Topic modeling is an unsupervised learning technique used to discover hidden topics in a collection of documents. It groups words into topics based on their co-occurrence patterns, helping to summarize and understand large datasets. Topic modeling is useful in document classification, summarization, and discovering thematic structures in text data.

31. **Explain the Latent Dirichlet Allocation (LDA) algorithm.**  
    LDA is a generative probabilistic model used for topic modeling. It assumes that documents are mixtures of topics, and each topic is a distribution over words. The algorithm works by iteratively assigning words to topics based on their likelihood within the document, then adjusting topic distributions to fit the data. This results in each document being represented by a distribution of topics, and each topic by a distribution of words.

   * Latent Dirichlet Allocation (LDA) is a powerful algorithm used in **topic modeling**, which helps uncover hidden topics within a collection of documents. Here's a more detailed explanation of how LDA works and its use cases:

   * How LDA Works:
      1. **Generative Model**: LDA assumes that documents are mixtures of different topics, and each topic is a distribution of words. For example, a news article may contain topics related to politics, sports, and technology, each contributing a certain proportion to the document.
      
      2. **Document Representation**: LDA represents each document as a combination of multiple topics. Each topic, in turn, is represented as a combination of words that are likely to appear together in that topic.
      
      3. **Probabilistic Process**: 
         - LDA assigns words to topics based on their probability of appearing in those topics.
         - It then iteratively adjusts these assignments to better fit the data.
         - This results in a distribution of topics for each document and a distribution of words for each topic.
      
      4. **Key Steps in LDA**:
         - **Initialization**: Randomly assign each word in the document to a topic.
         - **Gibbs Sampling**: Iteratively update the topic assignment for each word based on the likelihood of the word belonging to the current topic and the document’s topic distribution.
         - **Convergence**: After many iterations, the model converges, meaning the topic-word and document-topic distributions stabilize.
   
   * Use Cases of LDA:
   
      1. **Topic Discovery**: LDA is commonly used in applications where you want to discover the underlying themes in large collections of unstructured text, such as news articles, research papers, or social media posts. 
         - *Example*: Analyzing a large corpus of news articles to identify topics like "politics", "climate change", or "sports."
      
      2. **Document Classification**: Once the topics are identified, documents can be classified or grouped based on their topic distributions.
         - *Example*: Grouping customer reviews into categories such as "product quality", "customer service", or "pricing" based on the topics extracted by LDA.
      
      3. **Recommendation Systems**: LDA can be used to recommend content based on the topics a user is interested in.
         - *Example*: In a movie recommendation system, LDA can identify topics such as "action", "romance", or "comedy", and recommend movies that align with a user’s preferences based on their topic distribution.
      
      4. **Text Summarization**: By identifying the key topics within a document, LDA can be used to generate summaries that highlight the most important themes.
         - *Example*: Summarizing research papers or legal documents by extracting the primary topics discussed.
      
      5. **Sentiment and Opinion Mining**: LDA can help analyze topics associated with different sentiments in customer feedback or reviews.
         - *Example*: Identifying topics associated with positive or negative sentiments in online product reviews to understand customer opinions.

32. **Describe the bag-of-words (BoW) model.**  
   The Bag-of-Words (BoW) model is a simple representation of text where a document is converted into a vector of word frequencies or occurrences, disregarding word order and grammar. Each unique word in the document corpus becomes a feature, and the resulting vector captures the presence or count of those words in the document. BoW is often used in text classification and information retrieval tasks.

33. **What is dependency parsing?**  
   Dependency parsing is the process of analyzing the grammatical structure of a sentence by identifying the relationships (dependencies) between words. It creates a dependency tree where each node represents a word, and edges represent syntactic relationships, such as subject-verb or adjective-noun.

34. **How does dependency parsing differ from constituency parsing?**  
   Dependency parsing focuses on the relationships between words (dependencies), showing which words modify others. Constituency parsing, on the other hand, breaks a sentence into sub-phrases (constituents) based on its grammatical structure, producing a tree where phrases are nested within larger phrases (e.g., noun phrases or verb phrases). Dependency parsing is more concerned with word-to-word relations, while constituency parsing looks at phrase structure.

35. **Explain the concept of named entity recognition (NER).**  
   Named Entity Recognition (NER) is an NLP task that identifies and classifies named entities (like people, organizations, locations, dates) in text. It is crucial for information extraction tasks, helping to pull out key pieces of information for further analysis.

36. **What are some challenges faced in named entity recognition?**  
   Challenges in NER include:  
   - **Ambiguity**: A word may represent different entity types depending on context (e.g., "Apple" as a fruit vs. a company).
   - **Out-of-vocabulary (OOV) words**: Unseen entities can be hard to classify.
   - **Language-specific nuances**: Different languages require different NER approaches due to varying syntax, morphology, and cultural context.

37. **Describe the BIO tagging scheme used in NER.**  
   The BIO (Begin, Inside, Outside) tagging scheme is used to label tokens in NER tasks.  
   - **B-Tag** marks the beginning of an entity.
   - **I-Tag** indicates that the token is inside a named entity.
   - **O-Tag** is used for tokens that are outside any named entity.  
   This helps to distinguish between adjacent entities and correctly segment multi-word entities.

   Named Entity Recognition (NER) uses the **BIO tagging scheme** to label each token in a sentence.
   
      ### Example:
      Consider the sentence: "Apple Inc. was founded by Steve Jobs in Cupertino."
      
      - **"Apple"**: B-ORG (Beginning of an Organization)
      - **"Inc."**: I-ORG (Inside an Organization)
      - **"Steve"**: B-PER (Beginning of a Person)
      - **"Jobs"**: I-PER (Inside a Person)
      - **"Cupertino"**: B-LOC (Beginning of a Location)
      
      **Benefits**
      BIO tagging helps distinguish between different entities and identify multi-word entities accurately. For example, without a B-tag, the model could confuse "Steve Jobs" as two separate entities. The B and I tags ensure the model understands that they form a single entity (Person).
      
      ### How NER Models Use BIO Tagging:
      1. **Tokenization:** The text is first split into tokens (words or phrases).
      2. **Tagging:** Each token is assigned a tag (B, I, O) by the NER model.
      3. **Prediction:** Using machine learning or deep learning (e.g., CRF, LSTMs, Transformers), the model predicts which tokens belong to entities and labels them accordingly.
   
   This tagging scheme enables precise recognition of complex entities in text.

38. **What is sequence labeling, and why is it important in NLP?**  
   Sequence labeling is the task of assigning labels to each token in a sequence of text, such as words in a sentence. It is important in NLP for tasks like POS tagging, NER, and chunking, where the goal is to annotate text with meaningful tags that help in understanding its structure and content.

39. **Explain the concept of sequence-to-sequence learning.**  
   Sequence-to-sequence (seq2seq) learning is a type of model architecture used for tasks where input sequences (like sentences) are transformed into output sequences. It is widely used in machine translation, text summarization, and chatbot systems. Seq2seq models typically consist of an encoder that processes the input and a decoder that generates the output.

40. **What are some popular frameworks or libraries used in NLP?**  
   Popular frameworks and libraries in NLP include:
   - **NLTK**: For text processing and linguistic tasks.
   - **spaCy**: Efficient for production-grade NLP tasks with pre-trained models.
   - **Transformers (Hugging Face)**: For state-of-the-art models like BERT, GPT, etc.
   - **Gensim**: For topic modeling and word embeddings.
   - **StanfordNLP**: Known for deep learning-based NLP tasks.
   - **OpenNLP**: A machine learning toolkit for various NLP tasks.

41. **Describe some common evaluation metrics used in NLP tasks.**  
   Common evaluation metrics include:  
   - **Accuracy**: Measures the proportion of correct predictions, used in classification tasks like POS tagging.
   - **Precision, Recall, F1-score**: Precision measures the correctness of positive predictions, recall measures the ability to capture all relevant instances, and F1-score balances both. These are widely used in NER and text classification tasks.
   - **BLEU (Bilingual Evaluation Understudy)**: Used in machine translation to compare the similarity between machine-generated text and human reference translations.
   - **Perplexity**: Common in language modeling to evaluate how well a model predicts a sequence of words. Lower perplexity indicates better performance.
   - **ROUGE**: Used for summarization tasks to compare overlapping n-grams between generated and reference summaries.

42. **What is the BLEU score, and how is it used in NLP evaluation?**  
The BLEU (Bilingual Evaluation Understudy) score is a metric for evaluating the quality of text produced by machine translation models by comparing it to human reference translations. It measures precision of n-grams and incorporates penalties for overly short translations, providing a score between 0 and 1, where a higher score indicates better translation quality.

43. **Explain the concept of cross-entropy loss in NLP.**  
Cross-entropy loss is a measure used to evaluate the performance of classification models, including those in NLP. It quantifies the difference between the predicted probability distribution and the true distribution, penalizing incorrect predictions more heavily. It's commonly used in tasks like text classification and language modeling.

44. **How do you handle out-of-vocabulary words in NLP models?**  
Out-of-vocabulary (OOV) words are typically handled by using special tokens like `<UNK>` for unknown words. Techniques such as subword tokenization (e.g., Byte Pair Encoding) or character-level modeling can also help by breaking down OOV words into known subword units or characters.

45. **What is transfer learning, and how is it applied in NLP?**  
Transfer learning involves taking a pre-trained model on a large dataset and adapting it to a specific task with a smaller dataset. In NLP, this often means fine-tuning pre-trained language models (e.g., BERT, GPT) on task-specific data to leverage the general linguistic knowledge learned during pre-training.

46. **Describe some pre-trained language models, such as BERT, GPT, or RoBERTa.**  
BERT (Bidirectional Encoder Representations from Transformers) focuses on understanding context in both directions and is used for various NLP tasks. GPT (Generative Pre-trained Transformer) excels in generating coherent text by predicting the next word in a sequence. RoBERTa (A Robustly Optimized BERT Pretraining Approach) is a variant of BERT that improves performance by training with more data and longer sequences.

47. **How do you fine-tune a pre-trained language model for a specific task?**  
To fine-tune a pre-trained model, you initialize it with pre-trained weights and then train it on a task-specific dataset. This involves adjusting the model's parameters through additional training to adapt it to the new task, such as text classification or named entity recognition.

48. **What is text generation, and what are some challenges associated with it?**  
Text generation involves creating coherent and contextually relevant text based on a given input. Challenges include ensuring grammatical correctness, maintaining context over long passages, avoiding repetition, and generating content that aligns with specific styles or constraints.

49. **How do you deal with imbalanced datasets in NLP tasks?**  
To address imbalanced datasets, techniques such as oversampling the minority class, undersampling the majority class, using class weights, or employing data augmentation can be applied. Additionally, evaluation metrics like F1-score or balanced accuracy can provide a better measure of model performance on imbalanced data.

50. **Explain the concept of word sense disambiguation.**  
Word sense disambiguation (WSD) is the process of determining which meaning of a word is used in a given context when a word has multiple meanings. It involves analyzing the surrounding words and context to correctly identify the intended sense of the ambiguous word.

51. **What are some ethical considerations in NLP research and applications?**  
Ethical considerations include ensuring privacy and data security, avoiding biases in models that can lead to unfair treatment of certain groups, and being transparent about how models are trained and used. It's also important to consider the potential for misuse and to implement safeguards to mitigate harmful consequences.

      In the context of LLMs, **HHH** stands for **Honesty, Helpfulness, and Harmlessness**, key principles guiding responsible AI behavior:
   
      **Honesty**: The model should provide accurate, truthful, and reliable information, avoiding fabrication or misleading responses.
         
      **Helpfulness**: It should aim to assist users by offering relevant and constructive responses that address their queries or problems.
      
      **Harmlessness**: The model must avoid causing harm, such as by avoiding offensive, biased, or harmful language and ensuring it does not promote harmful actions or ideas.

52. **What types of tokenizers do you know? Compare them.**

- **Whitespace Tokenizers**: Split text on spaces. Simple but doesn't handle subwords well.
- **Character-level Tokenizers**: Break down text into individual characters. Useful in languages without clear word boundaries but produces long sequences.
- **Word-level Tokenizers**: Split text into words. Easy to understand, but struggles with rare or out-of-vocabulary (OOV) words.
- **Subword Tokenizers (WordPiece, BPE)**: Break text into subword units. Handle OOV words well and balance sequence length and vocabulary size. Used in modern transformer models.
  
53. **Can you extend a tokenizer? If yes, in what case would you do this? When would you retrain a tokenizer? What needs to be done when adding new tokens?**
Yes, tokenizers can be extended by adding new tokens (e.g., domain-specific words). You’d do this if you encounter many OOV words specific to your task (e.g., medical terms). You’d need to update the vocabulary and adjust tokenization rules. If many new tokens are added, you may need to retrain the tokenizer to learn optimal subword splitting. 

54. **How do regular tokens differ from special tokens?**
- **Regular tokens** represent standard words or subwords in a text.
- **Special tokens** are used for specific purposes, such as indicating sentence boundaries (`[CLS]`, `[SEP]` in BERT) or padding sequences (`[PAD]`). These help the model interpret and structure input correctly.

55. **Why is lemmatization not used in transformers? And why do we need tokens?**
Transformers don’t use lemmatization because tokenizers (like BPE or WordPiece) break text into subword units, allowing models to handle morphological variations inherently. Lemmatization is unnecessary because subwords provide enough flexibility. Tokens are crucial because transformers process sequences of token embeddings, not raw text.

56. **How is a tokenizer trained? Explain with examples of WordPiece and BPE.**
- **BPE (Byte-Pair Encoding)**: Starts with a base vocabulary of characters, then merges frequent character pairs iteratively to form subwords. Over time, common words are represented as single subwords, while rare ones are split into smaller units.
- **WordPiece**: Similar to BPE, but it prioritizes maximizing the likelihood of the training corpus given the tokenization. It chooses merges that help represent the corpus more efficiently. WordPiece is commonly used in BERT.

57. **What position does the CLS vector occupy? Why?**
The **CLS** (classification) token is the first token in the input sequence for models like BERT. It’s used to aggregate the information from the entire sequence because, after transformer layers, the hidden state of `[CLS]` is expected to represent the whole sequence, making it useful for classification tasks.

58. **What tokenizer is used in BERT, and which one in GPT?**
- **BERT** uses the **WordPiece** tokenizer.
- **GPT** models use **Byte-Pair Encoding (BPE)**.

59. **Explain how modern tokenizers handle out-of-vocabulary words?**
Modern tokenizers, like BPE and WordPiece, handle OOV words by breaking them into smaller subword units that are in the vocabulary. This ensures that even unseen words are represented by meaningful subcomponents.

60. **What does the tokenizer vocab size affect? How will you choose it in the case of new training?**
The **vocab size** affects model efficiency and coverage. A large vocabulary reduces the number of tokens per sentence but increases memory usage and complexity. A smaller vocabulary creates longer sequences but handles OOV words better. When training a new tokenizer, you’d choose the vocab size by balancing model size, sequence length, and the specific language/task needs.

61. **What is the difference between static and contextual embeddings?**
- **Static Embeddings**: Fixed word vectors that do not change with context (e.g., Word2Vec, GloVe). Each word has a single representation.
- **Contextual Embeddings**: Dynamic word vectors that vary depending on the surrounding words (e.g., ELMo, BERT). Each word’s representation changes based on its context in a sentence.

62. **What is class imbalance? How can it be identified? Name all approaches to solving this problem.**
**Class imbalance** occurs when some classes in a dataset are underrepresented compared to others. It can be identified by checking the distribution of class labels. **Approaches** to solve it include:
  - Resampling (oversampling minority class or undersampling majority class)
  - Using synthetic data generation (e.g., SMOTE)
  - Applying class weights in loss functions
  - Using anomaly detection techniques

63. **Can dropout be used during inference, and why?**
Dropout is not used during inference because it randomly drops units during training to prevent overfitting. During inference, dropout is disabled to ensure consistent and reliable predictions.

64. **What is the difference between the Adam optimizer and AdamW?**
**AdamW** differs from Adam by decoupling weight decay from the optimization step. AdamW applies weight decay directly to the weights, while Adam combines weight decay with the gradient updates.

65. **How do consumed resources change with gradient accumulation?**
**Gradient accumulation** reduces memory consumption by accumulating gradients over multiple mini-batches before performing a single update. This allows for larger effective batch sizes without requiring proportional memory.

66. **How to optimize resource consumption during training?**
Optimize resource consumption by:
  - Using gradient accumulation
  - Reducing batch size
  - Implementing mixed precision training
  - Employing efficient data loaders
  - Utilizing distributed training

67. **What ways of distributed training do you know?**
- **Data Parallelism**: Splitting data across multiple processors.
- **Model Parallelism**: Splitting the model across different processors.
- **Hybrid Parallelism**: Combining both data and model parallelism.
- **Distributed Data Parallel (DDP)**: Synchronizing gradients across multiple nodes.

68. **What is textual augmentation? Name all methods you know.**
**Textual augmentation** involves creating variations of text data to improve model robustness. Methods include:
  - Synonym replacement
  - Random insertion
  - Random deletion
  - Back-translation
  - Contextual augmentation (e.g., using models like BERT)

69. **Why is padding less frequently used? What is done instead?**
Padding is less frequently used because it can introduce inefficiencies. Instead, **dynamic batching** and **sequence bucketing** are used to handle varying sequence lengths more efficiently.

70. **Explain how warm-up works.**
**Warm-up** involves gradually increasing the learning rate from a small value to the target value over a few iterations at the beginning of training. This helps stabilize training and improve convergence.

71. **Explain the concept of gradient clipping?**
**Gradient clipping** involves setting a threshold to limit the magnitude of gradients during training. It prevents exploding gradients by clipping gradients that exceed a predefined value.

72. **How does teacher forcing work, provide examples?**
**Teacher forcing** involves using the true output from the training data as the next input during training a sequence model, rather than the model’s own previous prediction. For example, in sequence-to-sequence models, the true target word is fed as input at each step rather than the model’s predicted word.

73. **Why and how should skip connections be used?**
**Skip connections** help with training deep networks by allowing gradients to flow more easily through the network. They connect layers directly to later layers, bypassing intermediate layers to prevent vanishing gradients and improve learning.

74. **What are adapters? Where and how can we use them?**
**Adapters** are small trainable modules added to pre-trained models, allowing fine-tuning on specific tasks without modifying the entire model. They can be used to adapt models to new tasks with minimal training.

75. **Explain the concepts of metric learning. What approaches do you know?**
**Metric learning** focuses on learning a distance metric to measure similarity between data points. Approaches include:
  - **Contrastive Loss**: Minimizes the distance between similar pairs and maximizes it between dissimilar pairs.
  - **Triplet Loss**: Uses anchor, positive, and negative samples to enforce a margin between similar and dissimilar pairs.
  - **Siamese Networks**: Networks that learn embeddings by comparing pairs of inputs.


