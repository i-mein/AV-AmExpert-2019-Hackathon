![](/images/AmEx_logo.png)


# AV - AmExpert 2019 - Hackathon


AmExpert 2019: AV Hackathon by American Express [(Link)](http://datahack.analyticsvidhya.com/contest/amexpert-2019-machine-learning-hackathon/)


## Problem statement


Predict coupon redemption status for a credit card company (XYZ) to assist them in their discount marketing process using the power of machine learning.Based on previous transaction & performance data from the last 18 campaigns, predict the probability for the next 10 campaigns in the test set for each coupon and customer combination, whether the customer will redeem the coupon or not?





## Data description


The data available in this problem contains the following information: 

* User Demographic Details

* Campaign and coupon Details

* Product details

* Previous transactions




## Dataset schema

Here is the schema for the different data tables available. The detailed data dictionary is provided next.
![](/images/amex19_fig1.png)




* __train.csv:__ Train data


Variable | Definition
-------- | ----------
id	| Unique id for coupon customer impression
campaign_id	| Unique id for a discount campaign
coupon_id	| Unique id for a discount coupon
customer_id	| Unique id for a customer
redemption_status	(target) | (0 - Coupon not redeemed, 1 - Coupon redeemed)




* __campaign_data.csv:__ Campaign information for each of the 28 campaigns


Variable | Definition
-------- | ----------
campaign_id	| Unique id for a discount campaign
campaign_type	| Anonymised Campaign Type (X/Y)
start_date	| Campaign Start Date
end_date	| Campaign End Date




* __coupon_item_mapping.csv:__ Mapping of coupon and items valid for discount under that coupon


Variable |	Definition
-------- | ----------
coupon_id	| Unique id for a discount coupon (no order)
item_id |	Unique id for items for which given coupon is valid (no order)




* __customer_demographics.csv:__ Customer demographic information for some customers


Variable	| Definition
-------- | ----------
customer_id	| Unique id for a customer
age_range	| Age range of customer family in years
marital_status	| Married/Single
rented	| 0 - not rented accommodation, 1 - rented accommodation
family_size	| Number of family members
no_of_children	| Number of children in the family
income_bracket	| Label Encoded Income Bracket (Higher income corresponds to higher number)



* __customer_transaction_data.csv:__ Transaction data for all customers for duration of campaigns in the train data


Variable |	Definition
-------- | ----------
date	| Date of Transaction
customer_id	| Unique id for a customer
item_id |	Unique id for item
quantity |	quantity of item bought
selling_price	| Sales value of the transaction
other_discount	| Discount from other sources such as manufacturer coupon/loyalty card
coupon_discount	| Discount availed from retailer coupon



* __item_data.csv:__ Item information for each item sold by the retailer


Variable	| Definition
-------- | ----------
item_id	| Unique id for itemv
brand |	Unique id for item brand
brand_type |	Brand Type (local/Established)
category |	Item Category



* __test.csv:__ Contains the coupon customer combination for which redemption status is to be predicted


Variable	| Definition
-------- | ----------
id |	Unique id for coupon customer impression
campaign_id	| Unique id for a discount campaign
coupon_id |	Unique id for a discount coupon
customer_id	| Unique id for a customer


* __sample_submission.csv:__ This file contains the format in which you have to submit your predictions.



## Evaluation Metric

Submissions are evaluated on area under the ROC curve (AUC). 


## Leaderboard

- Public LB score (AUC): 0.8701    # Pos: 307

- Private LB score (AUC): 0.86197  # Pos: 268

* Submitted solution: XGB/LGB ensemble 


## Best scoring model: 

..to be upladed soon.. 










<!--  internal validation scheme AUC = 0.9213 -->


* Best scoring model (NOT submitted): AUC = 0.918   (to be uploaded soon..)

<!--  internal validation scheme AUC = 0.98 -->
