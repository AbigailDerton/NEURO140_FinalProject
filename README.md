# Predicting Customer Satisfaction using BERT, Domain Adaptability, and Extractive Summarization
### Abigail Derton

# 1. Introduction
Machine learning is one of the fastest-growing fields today with applications in medicine, facial recognition, language translation, self-driving cars, and beyond. Some subfields of machine learning, such as Computer Vision, have advanced rapidly due in part to transfer learning. Other subfields such as Natural Language Processing (NLP) have progressed at a slower rate due in part to the lack of large, labeled datasets which drive transfer learning methods. Another tricky aspect of deep learning NLP models is the computation time, which is positively associated with the length of training text. In this project, a Bidirectional Encoder Representations from Transformers (BERT) model is trained to predict star ratings from Amazon reviews. Domain adaptation is then used to predict star ratings from GoodReads book reviews. We hypothesize that the domain adaptation BERT model will perform just as well at predicting star ratings from book reviews as a model trained on the book review data. In addition, a BERT model is trained on a small subset of sentences from each review, chosen through extractive summarization, and compare prediction accuracy with that produced by the model trained on the full review texts.

# 2. Related work
Research has shown how the star rating of a product is significantly associated with the probability a customer will buy the product.1 With the abundance of online reviews through sites like Yelp, Amazon, ConsumerReports, and Tripadvisor, predicting customer satisfaction (i.e. star-rating) through text has become a popular machine learning task. Transformer-based models such as BERT have been shown to out-perform machine learning models at predicting star ratings from text reviews, with the caveat of higher computational cost.2,3,4 Recently, researchers used the BERT model and domain adaptation to predict positive (4-5 stars) and negative (1-2 stars) ratings from Amazon product reviews.5 In particular, these researchers trained multiple machine learning and deep learning models to predict positive and negative
ratings from Amazon clothing reviews and tested the accuracy of this model in predicting ratings from Amazon shoe reviews. They found that the BERT model produced the highest prediction accuracy on the target domain (shoe reviews), compared with other prediction models. To our knowledge, there is no current work showing how well a BERT model trained on Amazon reviews performs on Goodreads book reviews test data. Also to our knowledge, no current published work has explored domain adaptability for sentiment

# 3. Methods
## 3.1. Dataset
We use the Amazon Reviews dataset which is a collection of several million Amazon product reviews.6 Each review consists of written text and a star rating (0: 1-2 stars; 1: 4-5 stars). We also use the Goodreads Book Reviews dataset which consists of more than 15M reviews of approximately 2M books.7 Each review consists of written text along with a rating from 1 to 5.

We concatenated these star ratings to match those in the Amazon Reviews dataset (i.e. 0: 1-2 stars; 1: 4-5 stars).

## 3.2. BERT model
We used the BERT (Bidirectional Encoder Representations from Transformers) model to predict star ratings from the Amazon Reviews text.8 This pre-trained, transformer-based model uses encoder layers and self-attention heads to learn contextual relations between words.94,10 The BERT model is pre-trained on language modeling and next sentence prediction tasks, which allows it to perform well even when a limited amount of training data is available. For this study, we trained the BERT model in Python using the ktrain (v0.29.x) library, which is a wrapper for TensorFlow Keras. Code for the BERT model was taken from a ktrain tutorial available through GitHub and adapted to work with the Amazon and Goodreads datasets.11 The learning rate was optimized following a medium tutorial.12 Any target information present in the text reviews was removed before training or testing the models. Some reviews contained the star rating within the review text itself. For instance, one review starts with “I gave this product 2 stars because…”. To remove target information, we removed all digits and written digits (one, two, etc.) from the text. We also removed any occurrences of the words “rate” and “star” (including “rated”, “rating”, “stars”, etc.). Examples of the reviews before and after cleaning are given in the table below. The function to clean data was written by me and is available in my GitHub repository cited at the end of this document.13

|Original review|Cleaned review |
| -------------- | --------------- |
|“...since the bad guy at least fits what I expect this gets 2 stars.”| “...since the bad guy at least fits what I expect this gets.”|
|“...the reason it got 4 stars was for the lack of special features for both Spider-man and Spider-man 2.”|“...the reason it got was for the lack of special features for both Spider-man and Spider-man.”|
|“...I would have given the cable a 5 star rating, but the way I had to get the cable, by having to buy it separately, is a big rip off.”|“...I would have given the cable a, but the way I had to get the cable, by having to buy it separately, is a big rip off.”|

## 3.3. Transfer learning
One useful application of deep learning models is the prediction of customer satisfaction from written or verbal reviews of products. In some cases, the star ratings associated with reviews may be unavailable. In such cases, transfer learning can be used to predict customer satisfaction (i.e. star rating) on unlabeled data. In this study, a BERT model was trained on the Amazon Reviews dataset and its performance was tested on unseen Goodreads book reviews to predict the associated star ratings. This method is a form of transfer learning known as domain adaptation since model parameters trained on a source domain are transferred to perform the same task on an unseen target domain, by generalization.14 This allows for rapid prototyping and implementation to shorten the training time on the never before seen data given that the model is not relearning with every iteration by leveraging the pre-trained BERT model.

## 3.4. Extractive summarization
Extractive summarization is an NLP method that selects a prespecified number of sentences from a paragraph of text.15 Only sentences containing the most important information from the text are selected. In extractive summarization, vectors are created for each sentence in the text using the word tokens present in the sentence. Cosine similarity is then calculated between each sentence pair, and a similarity matrix is created. The sentences are then ranked based on similarity score and choose a prespecified number of sentences with the highest scores to be included in the summarized text. In this study, extractive summarization was used to shorten the length of reviews passed to the BERT model in an effort to reduce computational cost while maintaining high prediction accuracy.

|Source text|Extractive summary|
|-----------|------------------|
|“Glorious story: I loved Whisper of the wicked saints. The story was amazing and I was pleasantly surprised at the changes in the book. I am not normally someone who is into romance novels, but the world was raving about this book and so I bought it. I loved it !! This is a brilliant story because it is so true. This book was so wonderful that I have told all of my friends to read it. It is not a typical romance, it is so much more. Not reading this book is a crime, because you are missing out on a heart warming story..”|“The story was amazing and I was pleasantly surprised at the changes in the book. I loved it !! This is a brilliant story because it is so true”|

# 4. Results
Data analysis was performed on 50,000 Amazon reviews and 35,708 Goodreads reviews. Of the Amazon reviews, 25506 were negative (1-2 stars) and the remaining 24494 were positive (4-5 stars). Of the Goodreads reviews, 6616 were negative and 29093 were positive. Distributions of the sentence lengths for each dataset are shown in Figure 1. Boxplots of the lengths (#sentences) of positive and negative reviews for each dataset are shown in Figure 2.

![neuro140_fig1_fig2](https://github.com/user-attachments/assets/f5284173-a0b2-497c-ad58-b0fd879ba6dc)

An initial exploratory analysis was performed on a subset of 10K reviews from the Amazon Reviews dataset. The purpose of this analysis was to establish baseline prediction accuracy. In addition, extractive summarization was performed on this data subset to determine the minimum number of sentences required to maintain high prediction accuracy. The results of this analysis are summarized in Table 1. The model performed very well for the concatenated reviews. A model trained and tested on 1-sentence summaries of the reviews achieved a validation accuracy of 83%, while the model trained and tested on 2-sentence summaries achieved 90.7% validation accuracy. To account for the longer review lengths in the Goodreads dataset, reviews were concatenated to a maximum of three sentences for all models using concatenated reviews.

## Table 1: BERT Training and Prediction Accuracy Following Extractive Summarization on Subset of Amazon Reviews
|Maximum #Sentences|Training Loss|Training Accuracy|Validation Loss|Validation Accuracy|
|-----|-----|-----|-----|-----|
|1 |0.1740|0.9349|0.4374|0.8300|
|2 |0.0893|0.9713|0.2741|0.9070|
|3 |0.0564|0.9833|0.2715|0.9160|
|No maximum <br /> (full reviews)|0.0560|0.9836|0.2486|0.9400|

Six BERT models were implemented, as shown in Table 2. Even-numbered models (models 2, 4, and 6) were trained on the concatenated reviews with maximum lengths of three sentences. Odd-numbered models were trained on the full-length reviews. A maximum of 200 words from each review were passed to the models since more than 99.0% of the Amazon reviews and 73.5% of the Goodreads reviews had fewer than 200 words. The models each used a batch size of 16, a learning rate of , and 3 epochs. To account for the 2 * 10 unbalanced rating −5 distribution in the Goodreads dataset, positive reviews were undersampled in the training data. This resulted in training data of size 9312 consisting of 50% positive and 50% negative reviews. 

The training and validation accuracies of the six models are given in Table 3. The highest validation accuracy was achieved by Model 1 and the lowest validation accuracy was achieved by Model 4. The distribution of positive and negative reviews in the Goodreads dataset reflected that in the overall dataset (~81% positive). This presented a challenge for the models which were all trained on datasets with approximately 50% positive and 50% negative reviews. The unbalanced nature of the Goodreads dataset likely led to the low specificities shown in Table 4. The models trained on Amazon reviews performed the worst at correctly identifying negative Goodreads reviews.

Runtimes for training and testing each model were computed using the time library in python. These runtimes are given in Table 5. Concatenating the reviews improved training runtimes by roughly 30 seconds for each of the models. It also improved validation runtimes by between 3-8 seconds for each of the models.

## Table 2: BERT Model Descriptions
|Model 1 and Model 2 |BERT model trained and tested on 50K Amazon reviews, using a train-test-validation split of 70-15-15|
|---|---|
|Model 3 and Model 4 |BERT model trained and tested on ~36K Goodreads reviews, using a train-test-validation split of 70-15-15. Training data were oversampled so that 50% were negative|
|Model 5 and Model 6 |BERT model trained on 35K Amazon reviews and tested on ~5.4K Goodreads reviews|

## Table 3: BERT Training and Prediction Accuracy
<dl>
 <table>
    <tr>
      <td> </td>
      <td colspan="2">Full Text</td>
      <td colspan="2">Concatenated Text</td>
    </tr>
    <tr>
      <td> </td>
      <td >Training Accuracy</td>
      <td >Validation Accuracy</td>
      <td >Training Accuracy</td>
      <td >Validation Accuracy</td>
    </tr>
   <tr>
      <td>Trained and Tested on Amazon</td>
      <td >0.9869</td>
      <td >0.9487</td>
      <td >0.9780</td>
      <td >0.9239</td>
    </tr>
   <tr>
      <td>Trained and Tested on Goodreads</td>
      <td >0.9538</td>
      <td >0.8925</td>
      <td >0.9298</td>
      <td >0.8249</td>
    </tr>
   <tr>
      <td>Trained on Amazon and Tested on Goodreads</td>
      <td >0.9869</td>
      <td >0.8683</td>
      <td >0.9780</td>
      <td >0.8393</td>
    </tr>
  </table>
</dl>

## Table 4: BERT Validation Sensitivity and Specificity
<dl>
 <table>
    <tr>
      <td> </td>
      <td colspan="2">Full Text</td>
      <td colspan="2">Concatenated Text</td>
    </tr>
    <tr>
      <td> </td>
      <td >Sensitivity</td>
      <td >Specifity</td>
      <td >Sensitivity</td>
      <td >Specifity</td>
    </tr>
   <tr>
      <td>Trained and Tested on Amazon</td>
      <td >0.9594</td>
      <td >0.9378</td>
      <td >0.9304</td>
      <td >0.9170</td>
    </tr>
   <tr>
      <td>Trained and Tested on Goodreads</td>
      <td >0.8883</td>
      <td >0.8480</td>
      <td >0.8428</td>
      <td >0.7478</td>
    </tr>
   <tr>
      <td>Trained on Amazon and Tested on Goodreads</td>
      <td >0.9023</td>
      <td >0.6800</td>
      <td >0.8935</td>
      <td >0.6120</td>
    </tr>
  </table>
</dl>

## Table 5: Runtimes in Seconds for Training and Testing BERT Model
<dl>
 <table>
    <tr>
      <td> </td>
      <td colspan="2">Full Text</td>
      <td colspan="2">Concatenated Text</td>
    </tr>
    <tr>
      <td> </td>
      <td >Training Running Time</td>
      <td >Validation Running Time</td>
      <td >Training Running Time</td>
      <td >Validation Running Time</td>
    </tr>
   <tr>
      <td>Trained and Tested on Amazon</td>
      <td >3220.4</td>
      <td >93.4</td>
      <td >3190.4</td>
      <td >90.3</td>
    </tr>
   <tr>
      <td>Trained and Tested on Goodreads</td>
      <td >1027.1</td>
      <td >95.6</td>
      <td >1000.3</td>
      <td >87.8</td>
    </tr>
   <tr>
      <td>Trained on Amazon and Tested on Goodreads</td>
      <td >3220.4</td>
      <td >56.7</td>
      <td >3190.4</td>
      <td >53.2</td>
    </tr>
  </table>
</dl>

# 5. Conclusion
The BERT model achieved high overall accuracy for each classification task. However, specificity suffered when performing domain adaptation from the Amazon reviews to the Goodreads reviews. This contradicted our hypothesis that the domain adaptation BERT model would perform just as well at predicting star ratings from book reviews as a model trained on the book review data. Given that the specificities of Model 3 and Model 4 were significantly higher than Model 5 and Model 6, there may be a large difference in the language distributions between the negative Amazon and Goodreads reviews.

Models trained on extracted summaries of reviews achieved noticeably improved runtimes. In addition, the specificity and sensitivity decreased only slightly among the models that did not use domain adaptation. These results suggest that extractive summarization may be a useful tool for improving runtimes for BERT models when working with large datasets.

# 6. Challenges
One of the major challenges of transformer-based models is the computation time. The high computational cost of the BERT model limited the fine-tuning of parameters. For instance, even with the use of a GPU architecture, fine-tuning the learning rate of the model trained on 50,000 full Amazon reviews was estimated to take several days to run. Due to the time limits for GPU access in Google Colab, we used default BERT parameters rather than fine-tuned parameters.

7. Future Work
Future work will involve fine-tuning the parameters for each of the six BERT models. In addition, the models will be trained and tested at least three times each to find the average runtimes. Finally, the models will be re-trained multiple times using increasing amounts of data to estimate the relationship between dataset size and runtime improvements using extracted summaries of reviews.

## Works Cited:
1. Maslowska, E., Malthouse, E. C. & Bernritter, S. F. Too good to be true: the role of online
reviews’ features in probability to buy. Int. J. Advert. 36, 142–163 (2017).
2. Liu, Z. Yelp Review Rating Prediction: Machine Learning and Deep Learning Models.
(2020).
3. Comparing deep learning architectures for sentiment analysis on drug reviews. J. Biomed.
Inform. 110, 103539 (2020).
4. Taparia, A. & Bagla, T. Sentiment Analysis: Predicting Product Reviews’ Ratings using
Online Customer Reviews. (2020) doi:10.2139/ssrn.3655308.
5. Majumder, B. P., Mrini, K. & Shao, Y. Exploring Domain Adaptability for Sentiment
Classification Models. majumderb.com.
6. Bittlingmayer, A. Amazon Reviews for Sentiment Analysis.
7. UCSD Book Graph - Reviews.
https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/reviews.
8. Devlin, J., Chang, M.-W., Lee, K. & Toutanova, K. BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding. (2018) doi:10.48550/arXiv.1810.04805.
9. Hao, Y., Dong, L., Wei, F. & Xu, K. Visualizing and Understanding the Effectiveness of
BERT. in Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing and the 9th International Joint Conference on Natural Language Processing
(EMNLP-IJCNLP) 4143–4152 (2019).
10. Vaswani, A. et al. Attention is All you Need. (2017).
11. bhattbhavesh. ktrain-tutorial/ktrain-notebook.ipynb at main · bhattbhavesh91/ktrain-tutorial.
GitHub https://github.com/bhattbhavesh91/ktrain-tutorial.
12. Ahmed, H. BERT for Text Classification. Analytics Vidhya
https://medium.com/analytics-vidhya/finetuning-bert-using-ktrain-for-disaster-tweets-classifi
cation-18f64a50910b (2020).
13. AbbyDerton. NEURO140_FinalProject. GitHub
https://github.com/AbbyDerton/NEURO140_FinalProject.
14. Ben-David, S. et al. A theory of learning from different domains. Mach. Learn. 79, 151–175
(2010).
15. Page, L., Brin, S., Motwani, R. & Winograd, T. The PageRank Citation Ranking: Bringing
Order to the Web. (1999).

