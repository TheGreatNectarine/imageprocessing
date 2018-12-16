with open("data.txt", "r") as file:
    print(str([line.split("\t")[1] for line in file]).replace("[", "{").replace("]", "}").replace("\'", "\""))
