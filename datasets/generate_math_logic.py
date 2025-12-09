import csv
import random
import string


def generate_summation_dataset_csv(max_number, n=5000):
    with open(f"summation_{max_number}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])

        for i in range(n):
            a = random.randint(0, max_number)
            b = random.randint(0, max_number)

            if i % 2 == 0:
                correct_sum = a + b
                text = f"{a} + {b} = {correct_sum}"
                label = 1
            else:
                incorrect_sum = (
                    a + b +
                        random.choice([i for i in range(-10, 11) if i != 0])
                )
                text = f"{a} + {b} = {incorrect_sum}"
                label = 0

            writer.writerow([text, label])


def generate_inequality_dataset_csv(max_number, n=5000):
    with open(f"inequality_{max_number}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])

        for i in range(n):
            a = random.randint(0, max_number)
            b = random.randint(0, max_number)

            # 50% chance of being correct
            if i % 2 == 0:
                if a == b:
                    a += 1  # ensure inequality
                statement = f"{a} > {b}" if a > b else f"{b} > {a}"
                label = 1
            else:
                if a == b:
                    b += 1
                statement = f"{a} > {b}" if a <= b else f"{b} > {a}"
                label = 0

            writer.writerow([statement, label])


def generate_even_odd_dataset_csv(max_number, n=5000):
    with open(f"even_odd_{max_number}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])
        for i in range(n):
            a = random.randint(0, max_number)
            if i % 2 == 0:
                statement = f"{a if a % 2 == 0 else a + 1} is even"
                label = 1
            else:
                statement = f"{a if a % 2 != 0 else a + 1} is even"
                label = 0
            writer.writerow([statement, label])


def generate_divisibility_dataset_csv(max_number, divisor=5, n=5000):
    with open(f"divisible_by_{divisor}_{max_number}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])
        for i in range(n):
            if i % 2 == 0:
                a = random.randint(0, max_number // divisor) * divisor
                statement = f"{a} is divisible by {divisor}"
                label = 1
            else:
                a = random.randint(0, max_number)
                while a % divisor == 0:
                    a = random.randint(0, max_number)
                statement = f"{a} is divisible by {divisor}"
                label = 0
            writer.writerow([statement, label])


def generate_multiplication_dataset_csv(max_number, n=5000):
    with open(f"multiplication_{max_number}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])
        for i in range(n):
            a = random.randint(0, max_number)
            b = random.randint(0, max_number)
            if i % 2 == 0:
                correct = a * b
                statement = f"{a} * {b} = {correct}"
                label = 1
            else:
                incorrect = a * b + \
                    random.choice([j for j in range(-10, 11) if j != 0])
                statement = f"{a} * {b} = {incorrect}"
                label = 0
            writer.writerow([statement, label])


def generate_chained_inequality_dataset_csv(max_number, n=5000):
    with open(f"chained_inequality_{max_number}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])
        for i in range(n):
            if i % 2 == 0:
                a, b, c = sorted(random.sample(range(max_number), 3))
                statement = f"{a} < {b} < {c}"
                label = 1
            else:
                # force a false condition
                while True:
                    a = random.randint(0, max_number)
                    b = random.randint(0, max_number)
                    c = random.randint(0, max_number)
                    if not (a < b < c):
                        break
                statement = f"{a} < {b} < {c}"
                label = 0
            writer.writerow([statement, label])


def get_random_vars(n=2):
    """Get n unique random uppercase letters"""
    return random.sample(string.ascii_uppercase, n)


def generate_boolean_and_dataset_csv(n=5000):
    with open("boolean_and.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])
        for i in range(n):
            var1, var2 = get_random_vars(2)
            val1 = random.choice(["true", "false"])
            val2 = random.choice(["true", "false"])
            label = 1 if val1 == "true" and val2 == "true" else 0
            statement = f"If {var1} is {val1} and {var2} is {val2}, then {var1} and {var2} is true"
            writer.writerow([statement, label])


def generate_boolean_or_dataset_csv(n=5000):
    with open("boolean_or.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])
        for i in range(n):
            var1, var2 = get_random_vars(2)
            val1 = random.choice(["true", "false"])
            val2 = random.choice(["true", "false"])
            label = 1 if val1 == "true" or val2 == "true" else 0
            statement = f"If {var1} is {val1} and {var2} is {val2}, then {var1} or {var2} is true"
            writer.writerow([statement, label])


def generate_boolean_not_dataset_csv(n=5000):
    with open("boolean_not.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])
        for i in range(n):
            var = random.choice(string.ascii_uppercase)
            val = random.choice(["true", "false"])
            label = 1 if val == "false" else 0
            statement = f"If {var} is {val}, then NOT {var} is true"
            writer.writerow([statement, label])


def generate_boolean_xor_dataset_csv(n=5000):
    with open("boolean_xor.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])
        for i in range(n):
            var1, var2 = get_random_vars(2)
            val1 = random.choice(["true", "false"])
            val2 = random.choice(["true", "false"])
            label = 1 if (val1 == "true") != (val2 == "true") else 0
            statement = f"If {var1} is {val1} and {var2} is {val2}, then {var1} XOR {var2} is true"
            writer.writerow([statement, label])


def generate_boolean_implies_dataset_csv(n=5000):
    with open("boolean_implies.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])
        for i in range(n):
            var1, var2 = get_random_vars(2)
            val1 = random.choice(["true", "false"])
            val2 = random.choice(["true", "false"])
            label = 1 if val1 == "false" or val2 == "true" else 0
            statement = f"If {var1} is {val1} and {var2} is {val2}, then {var1} implies {var2} is true"
            writer.writerow([statement, label])


def generate_boolean_iff_dataset_csv(n=5000):
    with open("boolean_iff.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])
        for i in range(n):
            var1, var2 = get_random_vars(2)
            val1 = random.choice(["true", "false"])
            val2 = random.choice(["true", "false"])
            label = 1 if (val1 == "true") == (val2 == "true") else 0
            statement = f"If {var1} is {val1} and {var2} is {val2}, then {var1} if and only if {var2} is true"
            writer.writerow([statement, label])


def generate_boolean_nand_dataset_csv(n=5000):
    with open("boolean_nand.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])
        for i in range(n):
            var1, var2 = get_random_vars(2)
            val1 = random.choice(["true", "false"])
            val2 = random.choice(["true", "false"])
            label = 0 if val1 == "true" and val2 == "true" else 1
            statement = f"If {var1} is {val1} and {var2} is {val2}, then {var1} NAND {var2} is true"
            writer.writerow([statement, label])


def generate_boolean_nor_dataset_csv(n=5000):
    with open("boolean_nor.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])
        for i in range(n):
            var1, var2 = get_random_vars(2)
            val1 = random.choice(["true", "false"])
            val2 = random.choice(["true", "false"])
            label = 1 if val1 == "false" and val2 == "false" else 0
            statement = f"If {var1} is {val1} and {var2} is {val2}, then {var1} NOR {var2} is true"
            writer.writerow([statement, label])


def generate_digit_count_dataset_csv(n=5000):
    with open("digit_count.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])
        for i in range(n):
            num = random.randint(1, 99999)
            correct_len = len(str(num))
            if i % 2 == 0:
                statement = f"The number {num} has {correct_len} digits"
                label = 1
            else:
                incorrect_len = correct_len + random.choice([-2, -1, 1, 2])
                incorrect_len = max(1, incorrect_len)
                statement = f"The number {num} has {incorrect_len} digits"
                label = 0
            writer.writerow([statement, label])


def generate_set_membership_dataset_csv(max_number, n=5000):
    with open(f"set_membership_{max_number}.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])
        for i in range(n):
            the_set = sorted(random.sample(range(max_number), 5))
            if i % 2 == 0:
                x = random.choice(the_set)
                label = 1
            else:
                x = random.randint(0, max_number)
                while x in the_set:
                    x = random.randint(0, max_number)
                label = 0
            statement = f"{x} is in the set {the_set}"
            writer.writerow([statement, label])


generate_summation_dataset_csv(1000)
generate_summation_dataset_csv(10)
generate_inequality_dataset_csv(1000)
generate_inequality_dataset_csv(10)
generate_even_odd_dataset_csv(1000)
generate_even_odd_dataset_csv(10)
generate_divisibility_dataset_csv(1000, divisor=5)
generate_divisibility_dataset_csv(10, divisor=5)
generate_multiplication_dataset_csv(1000)
generate_multiplication_dataset_csv(10)
generate_chained_inequality_dataset_csv(1000)
generate_chained_inequality_dataset_csv(10)
generate_boolean_and_dataset_csv()
generate_boolean_or_dataset_csv()
generate_boolean_not_dataset_csv()
generate_boolean_xor_dataset_csv()
generate_boolean_implies_dataset_csv()
generate_boolean_iff_dataset_csv()
generate_boolean_nand_dataset_csv()
generate_boolean_nor_dataset_csv()
generate_digit_count_dataset_csv()
generate_set_membership_dataset_csv(10)
generate_set_membership_dataset_csv(1000)
