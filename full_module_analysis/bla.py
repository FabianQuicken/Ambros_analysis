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

person = "phd"
has_safety_training = False
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
