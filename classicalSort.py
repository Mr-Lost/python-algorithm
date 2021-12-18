import random


# 经典排序
class Sort:
    def __init__(self):
        self.list1 = [random.randint(0, 100) for _ in range(50)]

    # 选择排序：从头遍历找到最小值，依次与第1,2,3……位数字互换；时O(N^2)，空O(1)，不稳定
    def selection_sort(self):
        for i in range(len(self.list1)):
            temp = i
            for j in range(i+1, len(self.list1)):
                if self.list1[j] < self.list1[temp]:
                    temp = j
            self.list1[i], self.list1[temp] = self.list1[temp], self.list1[i]
        return self.list1

    # 插入排序：从头开始每次取后面的一个数跟前面所有数比较并插入(即维护一个有序的头部并依次拿下一个数插入)；时O(N^2)，空O(1)，稳定
    def insertion_sort(self):
        for i in range(1, len(self.list1)):
            for j in range(0, i):
                if self.list1[j] > self.list1[i]:
                    self.list1.insert(j, self.list1.pop(i))
        return self.list1

    # 冒泡排序：重复从头开始依次相邻两两比较并把较大值往后移；时O(N^2)，空O(1)，稳定
    def bubble_sort(self):  
        stop = len(self.list1)
        for i in range(len(self.list1)-1):
            for j in range(stop-1):
                if self.list1[j] > self.list1[j+1]:
                    self.list1[j], self.list1[j+1] = self.list1[j+1], self.list1[j]
            stop -= 1
        return self.list1

    # 归并排序：时O(N*logN)，空O(N)，稳定
    # 快速排序：时O(N*logN)，空O(logN)，不稳定
    # 堆排序：时O(N*logN)，空O(1)，不稳定

    # 基于比较的排序中：时间复杂度<O(N*logN)的排序还没有；时=O(N*logN)且空<O(N)且稳定的排序也还没有。


if __name__ == '__main__':
    lmr = Sort()
    print(lmr.selection_sort())
    print(lmr.bubble_sort() == sorted(lmr.list1))
