from collections import defaultdict

classes = defaultdict(list)

with open("data/wsj/wsj00-18.pos", mode="r") as f:
    for line in f.readlines():
        line = line.rstrip().split(" ")
        for ele in line:
            # print(ele)
            hoge = ele.split("/")
            if len(hoge) > 2:
                continue
            word, _class = ele.split("/")
            classes[_class].append(word)
    f.close()
with open("data/wsj/wsj19-21.pos", mode="r") as f:
    for line in f.readlines():
        line = line.rstrip().split(" ")
        for ele in line:
            # print(ele)
            hoge = ele.split("/")
            if len(hoge) > 2:
                continue
            word, _class = ele.split("/")
            classes[_class].append(word)
    f.close()

for key in classes.keys():
    count = 0
    print(key)
    with open(f"classes/{key}_words.txt", mode="w") as f:
        for ele in classes[key]:
            if count % 5 == 4:
                f.write(ele + "\n")
            else:
                f.write(ele + " ")
            count += 1
