# part1
def get_list(num):
    if num<=0:
        return []
    lst = []
    for i in range(num):
        lst.append(i)
    return lst

lst1 = get_list(0)
lst2 = get_list(10)

print(lst1)
print(lst2)

# 1:访问列表中值
print("get value by index -1: %d" % lst2[-1] )
print("get value by slicing [-1::-1]",lst2[-1::-1])

# 2:修改
lst1.append('a')
lst1.extend(lst2)
print("lst1: ",lst1)


# part2
# 1:+ * list()
s = "123456789"
print(list(s))

lst1.clear()
lst1+=list(s)
print("lst1.clear() + list(s): ",lst1)

lst1.clear()
lst1.append('a')
lst1*=10
print("lst1.clear() + append(a),then * 10: ",lst1)


# part3
lst1.clear()
print()
print("list2:",lst2)
print("len of lst2: %d" % len(lst2))
print(f"max of lst2: {max(lst2)}")
print(f"min of lst2: {min(lst2)}")

# part4
print()
print("for lst2")
lst2.append(10)
print(f"append 10: {lst2}")

lst1.append(11)
print(f"lst1: {lst1}")
lst2.extend(lst1)
print(f"extent lst1: {lst2}")

lst2.remove(1)
print(f"remove 1: {lst2}")

popped_valie = lst2.pop(-1)
print(f"pop(-1): {lst2}, popped_value: {popped_valie}")

print(f"count(5): {lst2.count(5)}")
print(f"index of elem-5: {lst2.index(5)}")

lst2.reverse()
print(f"reversed: {lst2}")

lst2.sort()
print(f"sorted: {lst2}")

lst2.insert(1,1)
print(f"insert 1 by index-1: {lst2}")