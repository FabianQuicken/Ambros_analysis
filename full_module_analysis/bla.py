"""
choice = "cappuccino"
milk_level = True
bean_level = 0.5

if choice == "cappuccino":
    if milk_level > 0.4 and bean_level > 0.2:
        print("Cappuccino in progress...")

time = 15

has_ring = False

if has_ring == False:
    print("Nooo")

person = "student"
has_safety_training = True
time = 22
is_emergency = False

if not is_emergency:
    if person == "pi":
        print("All hail the king")
    elif has_safety_training:
        if person == "phd" or person == "postdoc":
            print("Welcome working slave.")
        elif person == "student" and 18 >= time >= 8:
            print("Hello there, young padawan.")
        else:
            print("Access denied: Outside student hours.")
    else:
        print("Access denied: Safety training missing.")
else:
    print("Emergency access granted.")


import random
random_number = random.randint(1,10)
print(random_number)



shopping_list = ["banana", "apple", "bread", "meat", "milk"]
grocery_cart = []

for element in shopping_list:
    grocery_cart.append(element)
    #shopping_list.remove(element)
    print(shopping_list)
    print(grocery_cart)

"""
shopping_list = ["banana", "apple", "bread", "meat", "milk"]
grocery_cart = []



for i in range(len(shopping_list)):
    grocery_cart.append(shopping_list[-1])
    print(grocery_cart)
    shopping_list.pop(-1)
    print(shopping_list)

x = [4, 5, 6, 12, 3, 18, 15, 14, 9, 22, 2]

print(sorted(x, reverse=True))

students = {
	"Hannah": 3,
	"Fabi": 0,
	"Johanna": 4,
	"Yuliia": 2,
	"Moritz": 3,
	"Ilian": 6,
	"Christopher": 10,
	"Christoph": 1,
	"Lena": 2
}

for s in students:
    if students[s] >= 3:
        students[s] = False 
    else:
        students[s] = True

submitted = [
    "Hanna", # handed in first
    "Fabi",
    "Kevin",
    "Johanna",
    "Yuliia",
    "David",
    "Moritz",
    "Ilian",
    "Leonie",
    "Christopher",
    "Christoph",
    "Simon",
    "Lena" # handed in last
    ]

for i in range(1,20,2):
    print(i)
number_list = [2, 3, 4, 6,4, 7,3 ,4, 5]

for i in range(len(number_list)):

    print(i, number_list[i])

for i in range(5):
    print(i)

all_i = []
for i in range(1, 10, 2):
    all_i.append(i)

print(all_i)

number_list = [2, 5, 3, 10, 1, 4, 7]
for i, item in enumerate(number_list):
    if item > 5:
        print(i)

iterable = x
for item in iterable:
    # do this
    pass


a = 10
b = 5
if a != b:
    if a > b:
        print(f"{a} is the larger number.")
    else:
        print(f"{b} is the larger number.")
else:
    print("Both numbers are equal.")

choice = "Cappuccino"
milk_level = 0.7
bean_level = 0.5

if choice == "Cappuccino":
    if milk_level > 0.4 and bean_level > 0.2:
        print("Cappuccino in progress.")

weekend = False
holiday = True

if weekend or holiday:
    print("Store is closed.")

has_ring = False

if not has_ring:
    print("Stolen! My Precious!")