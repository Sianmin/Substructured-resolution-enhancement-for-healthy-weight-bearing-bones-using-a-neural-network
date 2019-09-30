import os
E = [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]]
displacement = [[1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]]
force = [[21, 10, 10]]
print(len(force))
def FEA_2D(file_name, E, displacement, force, fix_y=0):
    # 기초 데이터 생성
    [height, width] = E.shape
    ansys_input = open("%s.ans" % file_name ,'w')

    # Pre: Node 생성
    ansys_input.write("/PREP7\n"
                      "N, 1, 0, 0\n")
    ansys_input.write("NGEN, %d, 1, 1, 1, 1, 1, 0, 0\n" % width)
    ansys_input.write("NGEN, %d, 1, 1, 1, 1, 1, 0, 0\n" % height)

    # Pre: Element 생성 및 Material Property 부여
    ansys_input.write("ET, 1, PLANE182\n")
    for i in range(height):
        for j in range(width):
            ele = (i-1) * width + j
            n1 = (width+1) * (i-1) + j
            [n2, n3, n4] = [n1 + 1, n1 + width + 2, n1 + width + 1]

            ansys_input.write("MAT, %d\n\
            MP, PRXY, %d, 0.3\n\
            MP, EX, %d, %.1f\n" %\
                              (ele, ele, ele, E[i, j]))
            ansys_input.write("E, %d, %d, %d, %d\n" % (n1, n2, n3, n4))

    # Sol: B.C. & L.C.
    for i in range(len(displacement)):
        ansys_input.write("D, %d, UX, %.3f\n\
        D, %d, UY, %.3f\n" %\
                          (displacement[i, 1], displacement[i, 2]))
    for i in range(len(force)):
        ansys_input.write("F, %d, FX, %.3f\nF, %d, FY, %.3f\n" % (force[i, 1], force[i, 2]))
    if fix_y:
        ansys_input.write("NSEL, ALL\n"
                          "NSEL, R, LOC, Y, 0\n"
                          "D, ALL, UX, 0\n"
                          "D, ALL, UY, 0\n")
    ansys_input.write("ALLSEL\n"
                      "SOLVE\n"
                      "FINISH\n"
                      "/POST1\n")

    # Post: Displacement Field 구하기
    ansys_input.write("*DIM, U, array, %d, 2\n" % height * width)
    ansys_input.write("*DO, I, 1, %d\n\
    *GET, U(I, 1), NODE, I, U, X\n\
    *GET, U(I, 2), NODE, I, U, Y\n" \
                      % height * width)
    ansys_input.write("*MWRITE, U, %s_displacement, ans\n\
    (F15.8, F15.8)\n" % file_name)
    ansys_input.close()

    # Run FEA
    os.system ("ansys140 -b -np 2 -i %s.ans -o %s_output.ans" % (file_name, file_name))
