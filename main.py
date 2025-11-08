from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def getTrainingData():
    x_train = []
    y_train = []

    trainingData = open("training_data.txt", "rt")

    trainingDataContent = trainingData.read()
    trainingData.close()

    trainingDataContent = trainingDataContent.split("\n")

    for line in trainingDataContent:
        if line == "":
            continue

        trainingContent = line.split(" ")

        subject = int(trainingContent[len(trainingContent) - 1])

        trainingContent.pop()

        marks = [float(i) for i in trainingContent]

        x_train.append(marks)
        y_train.append(subject)

        # print(f"Marks: {marks} for subject: {subject}")
    
    x = np.array(x_train)
    y = np.array(y_train)
    
    return x, y

def getInputData():
    x_inputs = []

    inputData = open("alluvione_input_5.txt", "rt")

    inputDataContent = inputData.read()
    inputData.close()

    inputDataContent = inputDataContent.split("\n")

    for line in inputDataContent:
        if line == "":
            continue

        inputContent = line.split(" ")

        marks = [float(i) for i in inputContent]

        x_inputs.append(marks)

        # print(f"Marks: {marks} for subject: {subject}")

    return x_inputs

x, y = getTrainingData()
x_inputs = getInputData()

net = KNeighborsClassifier(n_neighbors=3)
net.fit(x, y.ravel())

def generateOutput():
    output_content = ""

    for x_input in x_inputs:
        prediction = net.predict(np.array(x_input).reshape(1, -1))
        
        output_content += f"{prediction[0]}\n"
    
    output_file = open("output.txt", "wt")
    output_file.write(output_content)
    output_file.close()

generateOutput()