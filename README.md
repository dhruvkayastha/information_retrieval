# Information Retrieval
Implementing Information Retrieval methods and the ICAIL '19 publication [Improving Sentence Retrieval from Case Law for Statutory Interpretation](https://doi.org/10.1145/3322640.3326736)

Data in the form of sentences for 3 retrieval queries, was obtained from the publication's github repo https://github.com/jsavelka/statutory_interpretation, which is a subset of [Caselaw access project](https://case.law/) data.

The sentences are classified into four categories according to their usefulness for the interpretation:

* **high value** - sentence intended to define or elaborate on the meaning of the term
* **certain value** - sentence that provides grounds to elaborate on the term's meaning
* **potential value** - sentence that provides additional information beyond what is known from the provision the term comes from
* **no value** - no additional information over what is known from the provision

Currently, the implementation includes the direct methods of retrieval - **Okapi BM25** and **TF-ISF**, which is a variant of **TF-IDF**. The Normalized Discounted Cumulative Gain (**NDCG**) has been used as the metric to measure ranking quality.
