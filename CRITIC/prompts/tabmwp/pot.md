Read the following table regarding "Coin collections" and then write Python code to answer a question:

Name | Number of coins
Braden | 76
Camilla | 94
Rick | 86
Mary | 84
Hector | 80
Devin | 83
Emily | 82
Avery | 87

Question: Some friends discussed the sizes of their coin collections. What is the mean of the numbers?
# Python Code, return answer
number_of_coins_for_different_person = [76, 94, 86, 84, 80, 83, 82, 87]
answer = sum(number_of_coins_for_different_person) / len(number_of_coins_for_different_person)

---

Read the following table regarding "" and then write Python code to answer a question:

Price | Quantity demanded | Quantity supplied
$155 | 22,600 | 5,800
$275 | 20,500 | 9,400
$395 | 18,400 | 13,000
$515 | 16,300 | 16,600
$635 | 14,200 | 20,200

Question: Look at the table. Then answer the question. At a price of $155, is there a shortage or a surplus? Choose from the the options: [shortage, surplus]
# Python Code, return answer
quantity_demanded_price_155 = 22600
quantity_supplied_price_155 = 5800
if quantity_demanded_at_price_155 > quantity_supplied_at_price_155:
    answer = 'shortage'
else:
    answer = 'surplus'

---

Read the following table regarding "Cans of food collected" and then write Python code to answer a question:

Samir | 7
Kristen | 4
Dakota | 7
Jamie | 8
Maggie | 9

Question: Samir's class recorded how many cans of food each student collected for their canned food drive. What is the median of the numbers?
# Python Code, return answer
cans = [7, 4, 5, 8, 9]
cans = sorted(cans)
middle1 = (len(cans) - 1) // 2
middle2 = len(cans) // 2
answer = (cans[middle1] + cans[middle2]) / 2

---

Read the following table regarding "" and then write Python code to answer a question:

toy boat | $5.54
toy guitar | $8.23
set of juggling balls | $5.01
trivia game | $8.18
jigsaw puzzle | $5.30
toy dinosaur | $3.00

Question: Lorenzo has $13.50. Does he have enough to buy a toy guitar and a set of juggling balls? Choose from the the options: ['yes', 'no']
# Python Code, return answer
guitar_price = 8.23
juggling_balls = 5.01
total_money = 13.5
if total_money > juggling_balls + guitar_price:
    answer = "yes"
else:
    answer = "no"

---