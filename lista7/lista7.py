# Questão 1
def find_judge(n, t):
    possible_judges = []
    for i in range(n):
        judge = True
        for j in range(n):
            if i is not j:
                if t[i][j] == 0:
                    judge = False
            else:
                if t[i][i] == 1:
                    judge = False
        if judge:
            possible_judges.append(i)
    if len(possible_judges) != 1:
        return -1
    return possible_judges[0]