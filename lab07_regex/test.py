import pandas as pd


# x = pd.DataFrame([['a', 'b'], ['a', 'a'], ['a', 'c']])
x = [['a', 'b'], ['a', 'a'], ['a', 'c']]
code_table = {"a": [1,0,0,0], "b": [0,1,0,0], "c":[0,0,1,0], "d": [0,0,0,1]}

print(x)
print()
print(code)
print()

# for i in range(x.shape[0]):
#     for j in range(x.shape[1]):
#         print(x.iloc[i][j])
#         print(code[x.iloc[i][j]])
#         x.iat[i][j] = code[x.iloc[i][j]]

for i in range(len(x)):
    for j in range(len(x[i])):
        x[i][j] = code[x[i][j]]

print(x)