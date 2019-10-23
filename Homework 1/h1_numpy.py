import re
import numpy as np

alfa = []
beta = []


def convert(matrix_a, matrix_x): #Transformare sub forma de matrice

    f = open("a.txt", "r")
    lines = f.readlines()
    f.close()

    for line in lines:
        words = re.split(" +", line)
        letters = [' ', ' ', ' ']
        operators = [' ', ' ', ' ']
        values = [0, 0, 0]
        for word in words:
            if word[len(word) - 1] in 'xyz':

                if word[len(word) - 1] == 'x':
                    letters[0] = 'x'
                elif word[len(word) - 1] == 'y':
                    letters[1] = 'y'
                elif word[len(word) - 1] == 'z':
                    letters[2] = 'z'

                if word[0] in '-+':  # cu operator la inceput (-8x, +4x)
                    if word[len(word) - 1] == 'x':
                        operators[0] = word[0]
                    elif word[len(word) - 1] == 'y':
                        operators[1] = word[0]
                    elif word[len(word) - 1] == 'z':
                        operators[2] = word[0]

                    nr = 0
                    for i in range(1, len(word) - 1):
                        nr *= 10
                        nr += int(word[i])

                    if nr == 0:
                        if word[len(word) - 1] == 'x':
                            values[0] = 1
                        elif word[len(word) - 1] == 'y':
                            values[1] = 1
                        elif word[len(word) - 1] == 'z':
                            values[2] = 1
                    else:
                        if word[len(word) - 1] == 'x':
                            values[0] = nr
                        elif word[len(word) - 1] == 'y':
                            values[1] = nr
                        elif word[len(word) - 1] == 'z':
                            values[2] = nr

                else:  # fara operator la inceput (8x, 4x)
                    nr = 0
                    for i in range(0, len(word) - 1):
                        nr *= 10
                        nr += int(word[i])

                    if nr == 0:
                        if word[len(word) - 1] == 'x':
                            values[0] = 1
                        elif word[len(word) - 1] == 'y':
                            values[1] = 1
                        elif word[len(word) - 1] == 'z':
                            values[2] = 1
                    else:
                        if word[len(word) - 1] == 'x':
                            values[0] = nr
                        elif word[len(word) - 1] == 'y':
                            values[1] = nr
                        elif word[len(word) - 1] == 'z':
                            values[2] = nr

            elif re.search('\d', word):  # verific daca ce e dupa egal e numar.
                matrix_x.append(int(word))

        row = [0, 0, 0]
        for i in letters:
            if i == 'x':
                if operators[0] == '-':
                    row[0] = -values[0]
                else:
                    row[0] = values[0]
            if i == 'y':
                if operators[1] == '-':
                    row[1] = -values[1]
                else:
                    row[1] = values[1]
            if i == 'z':
                if operators[2] == '-':
                    row[2] = -values[2]
                else:
                    row[2] = values[2]

        matrix_a.append(row)


convert(alfa, beta)

a = np.array(alfa)
b = np.array(beta)

print(f"Matricea A este egala cu: \n {a}")
print(f"Matricea B este egala cu: \n {b}")

det = np.linalg.det(a)
if int(det) == 0:
    print("Determinantul este egal cu 0.")
elif int(det) < 0:
    print(f"Nu exista solutii deoarece determinantul este mai mic decat 0.")
else:
    print(f"Determinantul matricei A este egal cu:\n {int(det)}")

    transpose_matrix = np.transpose(a)
    print(f"Matricea transpusa a lui A este: \n {transpose_matrix}")

    cofactor_matrix = np.linalg.inv(transpose_matrix) * det
    print(f"Matricea A* este egala cu: \n {cofactor_matrix}")

    inverse_matrix = np.linalg.inv(a)
    print(f"Inversa matricei A este egala cu: \n {inverse_matrix}")

    solution = np.linalg.solve(a, b)
    print(f"Solutia ecuatiei este egala cu: \n {solution}")


