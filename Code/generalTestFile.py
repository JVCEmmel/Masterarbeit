class Example:
    def __init__(self, count):
        self.count = count

examples = []
count_list =[]

for count in range(10):
    example = Example(count)

    examples.append(example)
    count_list.append(count)

    print(example.count)

    if 3 not in count_list:
        print("Hier ist keine drei drin")