# DACON_Code_Similarity-Contest
## Introduction
I was in 8th place in the Code Similarity Contest held at DACON.
I focused on the augmentation method.
The data merged with the existing data and the output from those data was used as the Augmentation method.

## File Description
list-comprehension folder : I tried to use it as additional data to improve the understanding of list comprehensions for model.
                            but i didn't use.<br>
augmentation_method 1& 2: The code for the augmentation method.<br>
code_funtion.py: Functions are used to construct dataset.<br>
p2top3_test.ipynb: The code changed the python2 code in the test dataset to python 3 code.<br>
generate_dataset: The code create a dataset that combines Origin data and Augmentation data.<br>
train&submit: train&submit code. 

## Result
Using outputs of code as augmantion data prove to improve score <br>
I used 2 different models. graphcodebert and unix.<br>
graphcodebert's max length is 512, unix's max length is 1024.<br>
unix has better score. <br>
The final step, mixing the results of two different models, dramatically increases the score.<br>
graphcodebert -> 0.968<br>
unix -> 0.972<br>
ensemble -> 0.976

## LISENSE
MIT License
