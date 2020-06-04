import random


# 经典排序
class Sort:
    def __init__(self):
        self.list1 = [random.randint(0, 100) for i in range(10)]

    # 直接插排：从头开始每次取后面的一个数依次跟前面所有数比较并插入；稳定
    def straight_insertion_sort(self):   
        for i in range(1, len(self.list1)):
            for j in range(0, i):
                if self.list1[j] > self.list1[i]:
                    self.list1.insert(j, self.list1[i])
                    self.list1.pop(i+1)
        return self.list1

    # 冒泡排序：重复从头开始两两比较并把较大值往后移；稳定
    def bubble_sort(self):  
        le = len(self.list1)
        for i in range(len(self.list1)-1):
            for j in range(le-1):
                if self.list1[j] > self.list1[j+1]:
                    self.list1[j], self.list1[j+1] = self.list1[j+1], self.list1[j]
            le -= 1
        return self.list1


if __name__ == '__main__':
    print(Sort().straight_insertion_sort())
