
element0 = {"Class": "74", "Class_name": "person", "Score": 1}
element1 = {"Class":"65", "Class_name": "horse", "Score": -1}

elements = {"element0": element0, "element1": element1}
to_be_deleted = {}
for element in elements:
    if elements[element]["Score"] < 0:
        print(elements[element]["Score"])
        to_be_deleted[element].append(element)

[elements.pop(item) for item in to_be_deleted]
print(to_be_deleted)
print(len(elements))
    