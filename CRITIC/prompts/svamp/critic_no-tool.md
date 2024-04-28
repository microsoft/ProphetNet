---

Question: Marco and his dad went strawberry picking. Marco's dad's strawberries weighed 11 pounds. If together their strawberries weighed 30 pounds. How much did Marco's strawberries weigh?
```python
num_strawberries_dad_picked = 11
num_strawberries_marco_picked = 30
answer = num_strawberries_dad_picked + num_strawberries_marco_picked
```

What's the problem with the above code?

1. The weight of straberries should be positive, 41.0 > 0, it's reasonable.
2. Let's check the code:

> answer = num_strawberries_dad_picked + num_strawberries_marco_picked

The above code calculates the total weight of strawberries picked by both Marco and his dad, instead of finding the weight of strawberries picked by Marco alone.

According to the question, the total weight of strawberries picked by both Marco and his dad is 30 pounds, and Marco's dad's strawberries weighed 11 pounds. To find the weight of Marco's strawberries, we need to subtract the weight of his dad's strawberries from the total weight.

Here's a better solution:
```python
total_weight = 30
dad_weight = 11

# Calculate the weight of Marco's strawberries
marco_weight = total_weight - dad_weight
answer = marco_weight
```

---

Question: Zachary did 46 push-ups and 58 crunches in gym class today. David did 38 more push-ups but 62 less crunches than zachary. How many more crunches than push-ups did Zachary do?
```python
zachary_push_ups = 46
zachary_crunches = 58
zachary_diff = zachary_crunches - zachary_push_ups
answer = zachary_diff
```

What's the problem with the above code?

1. The answer, 12, is a reasonable result.
2. Let's check the code:

> zachary_diff = zachary_crunches - zachary_push_ups

The code calculates the difference between the number of crunches and push-ups that Zachary did, it is correct.

The information about David is not relevant to the question and does not need to be considered in the calculation. There is no problem with the above code. 

Here's a better solution:
```python
zachary_push_ups = 46
zachary_crunches = 58

# Calculate the difference between Zachary's crunches and push-ups
zachary_diff = zachary_crunches - zachary_push_ups
answer = zachary_diff
```

---

Question: Faye had 35 packs of pencils each one having 4 pencils. She was placing her pencils into rows with 2 pencils in each row. How many rows could she make?
```python
num_packs_of_pencils = 35
num_pencils_in_each_pack = 4
answer = num_packs_of_pencils * num_pencils_in_each_pack
```

What's the problem with the above code?

1. The answer rows 140 > 0, it's reasonable.
2. Let's check the code:

> answer = num_packs_of_pencils * num_pencils_in_each_pack

The code calculates the total number of pencils Faye has, but it does not calculate the number of rows she can make. It needs to be modified to take into account the fact that she is placing 2 pencils in each row.

Here's a better solution:
```python
num_packs_of_pencils = 35
num_pencils_in_each_pack = 4

# Calculate the total number of pencils
total_pencils = num_packs_of_pencils * num_pencils_in_each_pack

# Calculate the number of rows with 2 pencils in each row
num_pencils_in_each_row = 2
num_rows = total_pencils // num_pencils_in_each_row
answer = num_rows
```

---

Question: Jack received 4 emails and sent 2 emails in the morning. He then received 6 emails and sent 8 emails in the afternoon. How many emails did Jack send in the day?
```python
num_emails_received_in_morning = 4
num_emails_sent_in_morning = 2
num_emails_received_in_afternoon = 6
num_emails_sent_in_afternoon = 8
answer = num_emails_sent_in_morning + num_emails_sent_in_afternoon
```

What's the problem with the above code?

1. The answer, 10, is a reasonable result.
2. Let's check the code:

> answer = num_emails_sent_in_morning + num_emails_sent_in_afternoon

The code calculates the total number of emails Jack sent during the day by adding the number of emails he sent in the morning to the number of emails he sent in the afternoon. There is no problem with the above code. 

Here's a better solution:
```python
num_emails_received_in_morning = 4
num_emails_sent_in_morning = 2
num_emails_received_in_afternoon = 6
num_emails_sent_in_afternoon = 8
answer = num_emails_sent_in_morning + num_emails_sent_in_afternoon
```

---