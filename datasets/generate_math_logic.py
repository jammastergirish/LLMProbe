import csv
import random


def generate_arithmetic_dataset_csv(max_number, n=5000):
    with open(f"arithmetic_{max_number}.csv", mode="w", newline="") as file:
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
                incorrect_sum = a + b + \
                    random.choice([i for i in range(-10, 11) if i != 0])
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


# Example usage




# 1. Even or Odd


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

# 2. Divisibility


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

# 3. Multiplication


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

# 4. Chained Inequality


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

# 5. Logical AND Truth Table


def generate_logical_and_dataset_csv(n=5000):
    with open("logical_and.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["statement", "label"])
        for i in range(n):
            A = random.choice(["true", "false"])
            B = random.choice(["true", "false"])
            label = 1 if A == "true" and B == "true" else 0
            statement = f"If A is {A} and B is {B}, then A and B is true"
            writer.writerow([statement, label])

# 6. Digit Counting


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

# 7. Set Membership


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


generate_arithmetic_dataset_csv(1000)
generate_arithmetic_dataset_csv(10)
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
generate_logical_and_dataset_csv()
generate_digit_count_dataset_csv()
generate_set_membership_dataset_csv(10)
generate_set_membership_dataset_csv(1000)
