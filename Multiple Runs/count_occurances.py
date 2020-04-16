if __name__ == '__main__':

    fileRead = open("progress.txt", "r")
    counts = [0, 0, 0, 0, 0]
    words = ["good", "happy", "afraid", "red", "blue"]
    accuracy = 0
    valuesC = []
    averageC = 0
    runs_remaining = 0

    for line in fileRead:
        if "runs are left" in line:
            runs_remaining = int(line[0:2])
        if "Highest C:" in line:
            acc = float(line[10:])
            valuesC.append(acc)

            if averageC:
                averageC = (acc + averageC) / 2
            else:
                averageC = acc

        if "Training accuracy:" in line:
            if accuracy:
                accuracy = (float(line[18:].replace("\n", "")) + accuracy)/2
            else:
                accuracy = float(line[20:].replace("\n", ""))
        for pos in range(len(words)):
            if words[pos] in line and "PASSED" in line:
                counts[pos] += 1

    print("Out of", (100 - runs_remaining), "runs:")
    for pos in range(len(words)):
        print(words[pos], "PASSED", counts[pos], "times")

    print("Average accuracy:", accuracy)
    print("All C values:", valuesC)

    maxC = 0
    minC = 999999
    averageC = 0

    for C in valuesC:
        minC = C if C < minC else minC
        maxC = C if C > maxC else maxC
        averageC += C

    print("minC:", minC)
    print("maxC:", maxC)
    print("averageC:", averageC/len(valuesC))

