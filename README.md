We will use itemeset mining for building features for classification. We will use the “Congressional Voting Records Data Set” for this task also. Our task is to convert the dataset into an itemset dataset and then to extract classification association rule. You can make this conversion by using the following simple python script.

Output of the above program is an itemset dataset. In this dataset we have listed each record  as itemset of the ’y’ valued features. Each feature is represented with an integer between 1 and
16.	We have also make the class label as one of the items (for republican we  have  used 100,  for democratic we have used 200). You can use Eclat algorithm for finding frequent itemset from this dataset. Answer the following questions:

a.	Run the itemset mining algorithm with 20% support. How many frequent itemsets are there?

b.	Write top 10 itemsets (in terms of highest support value).

c.	How many frequent itemsets have 100 as part of itemsets?

d.	How many frequent itemsets have 200 as part of itemsets?

e.	Write top 10 association rules (in terms of highest confidence value) where the rule’s head is 100

f.	How many rules with head 100 are there for which the confidence value is more than 75%? List them. For this you need to write a small script that finds confidence of a rule from the support value of its body and the support value of its body plus head.

g.	Write top 10 association rules (in terms of highest confidence value) where the rule’s head is 200

h.	How many rules with head 200 are there for which the confidence value is more than 75%? List them.

i.	Use the rules (which has more than 75% confidence) as binary feature and construct a new dataset, in which each rule is a feature and any transaction that has the body of the rule will have a feature value of 1 and if it does not have the body of the rule will have a feature value
0. Use soft-margin SVM with linear kernel to report 3-fold classification accuracy (after using 1 fold for parameter tuning). Report both average and standard deviation.
