import re

a = []
b = []


def convert(matrix_a, matrix_x):  # Transformare sub forma de matrice

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


def determinant_value(matrix_a):
    return matrix_a[0][0] * ((matrix_a[1][1] * matrix_a[2][2]) + (matrix_a[1][2] * (-matrix_a[2][1]))) \
           - matrix_a[1][0] * ((matrix_a[0][1] * matrix_a[2][2]) + (matrix_a[0][2] * (-matrix_a[2][1]))) \
           + matrix_a[2][0] * ((matrix_a[0][1] * matrix_a[1][2]) + (matrix_a[1][1] * (-matrix_a[0][2])))


def transpose_matrix(matrix_a):
    matrix_a[0][1], matrix_a[1][0] = matrix_a[1][0], matrix_a[0][1]
    matrix_a[0][2], matrix_a[2][0] = matrix_a[2][0], matrix_a[0][2]
    matrix_a[1][2], matrix_a[2][1] = matrix_a[2][1], matrix_a[1][2]
    return matrix_a


def cofactor_matrix(trans_a):
    b11 = (trans_a[1][1] * trans_a[2][2]) + (trans_a[1][2] * -trans_a[2][1])
    b12 = -((trans_a[1][0] * trans_a[2][2]) + (trans_a[2][0] * -trans_a[1][2]))
    b13 = (trans_a[1][0] * trans_a[2][1]) + (trans_a[2][0] * -trans_a[1][1])
    b21 = -((trans_a[0][1] * trans_a[2][2]) + (trans_a[2][1] * -trans_a[0][2]))
    b22 = (trans_a[0][0] * trans_a[2][2]) + (trans_a[2][0] * -trans_a[0][2])
    b23 = -((trans_a[0][0] * trans_a[2][1]) + (trans_a[2][0] * -trans_a[0][1]))
    b31 = (trans_a[0][1] * trans_a[1][2]) + (trans_a[1][1] * -trans_a[0][2])
    b32 = -((trans_a[0][0] * trans_a[1][2]) + (trans_a[1][0] * -trans_a[0][2]))
    b33 = (trans_a[0][0] * trans_a[1][1]) + (trans_a[1][0] * -trans_a[0][1])
    c_matrix = [[b11, b12, b13],
                [b21, b22, b23],
                [b31, b32, b33]]
    return c_matrix


def inverse_matrix(c_matrix, det_value):
    i_matrix = []
    value = 1/det_value
    for i in c_matrix:
        row = []
        for j in i:
            j *= value
            row.append(j)
        i_matrix.append(row)

    return i_matrix


def get_final_result(i_matrix, matrix_b):
    res = []
    for i in i_matrix:
        cnt = 0
        s = 0
        for j in i:
            s += j * matrix_b[cnt]
            cnt += 1
        res.append(s)
    return res


convert(a, b)
print(f"Matricea A este egala cu: \n {a}")
print(f"Matricea B este egala cu: \n {b}")

det = determinant_value(a)
if det == 0:
    print(f"Determinantul este egal cu {det}.")
elif det < 0:
    print(f"Nu exista solutii deoarece determinantul este mai mic decat 0.")
else:
    print(f"Determinantul matricei A este egal cu:\n {det}")
    transpose_a = a
    transpose_a = transpose_matrix(transpose_a)
    print(f"Transpusa matricei A este egala cu: \n {transpose_a}")

    co_matrix = cofactor_matrix(transpose_a)
    print(f"A* este egal cu: \n {co_matrix}")

    inv_matrix = inverse_matrix(co_matrix, det)
    print(f"Inversa matricei este egala cu: \n {inv_matrix}")

    x = get_final_result(inv_matrix, b)  # Matricea X
    print(f"Solutia ecuatiei este: \n {x}")
