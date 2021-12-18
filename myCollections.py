import random
from queue import PriorityQueue as pQueue


class Node:
    def __init__(self, value=None, nex=None):
        self.value = value
        self.next = nex


class BinaryTree:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


class MyCollections:
    def maxJoint(self, cards):
        # 所有数能拼接成的最大数
        # cards = ['234', '51', '200', '9', '90', '91', '900', '4001']
        cards = list(map(str, cards))
        cards.sort(key=lambda x: len(x))
        length = len(cards[-1])
        print(length)
        # cards.sort(key=lambda y: y.ljust(length, '9'))   # 拼成最小数
        cards.sort(key=lambda x: x+'9'*(length-len(x)), reverse=True)
        print(cards)
        print(''.join(cards))

    def nineToTen(self, n):
        # 9进制转10进制(例如：抹去4)
        # n = 523
        n = str(n)
        ans = 0
        for i, j in enumerate(n):
            if j > '4':
                ans += (ord(j)-ord('1')) * 9**(len(n)-i-1)
            else:
                ans += (ord(j)-ord('0')) * 9**(len(n)-i-1)
        print(ans)

    def findOnlyOneOddTimes(self, nums):
        # 找出数组中唯一的那个出现奇数次的数
        eor = 0                         # a^a=0, 0^a=a, ^不进位相加，满足交换，结合律
        for num in nums:
            eor ^= num
        print(eor)

    def findOnlyTwoOddTimes(self, nums):
        # 数组中有唯二个出现奇数次的数
        eor1 = 0
        for num in nums:
            eor1 ^= num                 # eor1 = a^b，eor1 != 0，且必有一位为1
        eorr = eor1 & (~eor1+1)         # 找到eor1最右边那个1，在这一位上，a,b一定一个为0，一个为1
        eor2 = 0
        for num in nums:
            if num & eorr == 0:
                eor2 ^= num             # 将eorr中的那位为1的所有数^起来，则结果为a,b中这位为1的其中一个
        eor1 ^= eor2
        print(eor1, eor2)

    def maxItem(self, nums, left, right):
        # 递归求数组中最大数
        if left == right:
            return nums[left]
        mid = left + ((right-left) >> 1)    # 防止 left+right 溢出
        maxLeft = self.maxItem(nums, left, mid)
        maxRight = self.maxItem(nums, mid+1, right)
        return maxLeft if maxLeft >= maxRight else maxRight

    def mergeSort(self, nums, left, right):
        # 归并排序，时间复杂度O(N*logN)，空间复杂度O(N)
        def merge(nums, l, m, r):
            temp = []
            p1 = l
            p2 = m+1
            while p1 <= m and p2 <= r:
                if nums[p1] <= nums[p2]:
                    temp.append(nums[p1])
                    p1 += 1
                else:
                    temp.append(nums[p2])
                    p2 += 1
            temp += nums[p1:m+1] if p1 <= m else nums[p2:r+1]
            for i in range(len(temp)):
                nums[l+i] = temp[i]

        if left == right:
            return
        mid = left + ((right - left) >> 1)
        self.mergeSort(nums, left, mid)
        self.mergeSort(nums, mid + 1, right)
        merge(nums, left, mid, right)
        return nums

    def smallSum(self, nums):
        # 当前项左边小于该数的所有数的总和，再总和
        def process(nums, l, r):
            if l == r:
                return 0
            m = l + ((r-l) >> 1)
            return process(nums, l, m) + process(nums, m+1, r) + merge(nums, l, m, r)

        def merge(nums, l, m, r):
            if l == r:
                return
            temp = []
            ans = 0
            p1 = l
            p2 = m+1
            while p1 <= m and p2 <= r:
                if nums[p1] < nums[p2]:             # 两边数相等要先append右边的数
                    temp.append(nums[p1])
                    ans += nums[p1] * (r-p2+1)
                    p1 += 1
                else:
                    temp.append(nums[p2])
                    p2 += 1
            temp += nums[p1:m+1] if p1 <= m else nums[p2:r+1]
            for i in range(len(temp)):
                nums[l+i] = temp[i]
            return ans

        if len(nums) < 2:
            return 0
        return process(nums, 0, len(nums)-1)

    def quickSort(self, nums, left, right):
        # 快排，由概率计算得出时间复杂度O(N*logN)，空间复杂度O(logN)
        def partition(nums, l, r):      # 小于在左，等于在中，大于在右
            ll = l-1                    # 小于区域的右边界
            rr = r                      # 大于区域的左边界
            while ll < rr and l < rr:           # 当两边界未重叠且当前指针未进入大于区域时，l为当前遍历指针，r为最后一项作为参照数的指针
                if nums[l] < nums[r]:           # 当前项 < 参照项，小于区域向右扩充一位，并将扩充项与当前项交换，当前指针l右移即指向下一项
                    ll += 1
                    nums[l], nums[ll] = nums[ll], nums[l]
                    l += 1
                elif nums[l] == nums[r]:        # 当前项 = 参照项，仅当前指针l右移
                    l += 1
                else:                           # 当前项 > 参照项，大于区域向左扩充一位，扩充项与当前项交换，当前指针不变
                    rr -= 1
                    nums[l], nums[rr] = nums[rr], nums[l]
            nums[rr], nums[r] = nums[r], nums[rr]
            return [ll+1, rr]          # 返回等于区域的左右边界

        if left < right:
            t = random.randint(0, right-left)
            nums[left+t], nums[right] = nums[right], nums[left+t]
            p = partition(nums, left, right)
            self.quickSort(nums, left, p[0]-1)
            self.quickSort(nums, p[1]+1, right)
            return nums

    def heapInsert(self, nums, index):
        # 堆结构就是用数组实现的完全二叉树结构。优先级队列就是堆结构
        # 大根堆：完全二叉树中所有子树的最大值都在顶部。小根堆反之
        # 当前节点为i，左子2i+1，右子2i+2，父(i-1)//2
        while index > 0 and nums[index] > nums[(index-1) >> 1]:
            p = (index-1) >> 1
            nums[index], nums[p] = nums[p], nums[index]
            index = p

    def heapify(self, nums, index, heapSize):
        # 在index位置的数是否需要下移以保持整体大根堆。heapSize堆的长度
        # 时间复杂度O(logN)
        left = index*2+1
        while left < heapSize:      # 有孩子
            largest = left+1 if left+1 < heapSize and nums[left] < nums[left+1] else left   # 返回孩子中的较大
            largest = largest if nums[largest] > nums[index] else index                     # 返回大孩子和父节点的较大
            if largest == index:        # 父节点最大即满足了大根堆，停止
                break
            nums[largest], nums[index] = nums[index], nums[largest]
            index = largest
            left = index*2+1

    def heapSort(self, nums):
        # 堆排序，时间复杂度O(N*logN)，空间复杂度O(1)
        if len(nums) < 2:
            return nums
        heapSize = len(nums)
        for index in range(heapSize):
            self.heapInsert(nums, index)
        # 使用heapify代替heapInsert速度会快些
        # for j in range(heapSize-1, -1, -1):
        #     self.heapify(nums, j, heapSize)
        while heapSize:
            heapSize -= 1
            nums[0], nums[heapSize] = nums[heapSize], nums[0]
            self.heapify(nums, 0, heapSize)
        return nums

    def distanceLessThanKSort(self, nums, k):
        # 一个几乎排好序的数组中，每一位数的位置和排好序后的位置不会偏移超过k
        heap = pQueue()
        k = k if k < len(nums) else len(nums)-1
        index = 0
        for i in range(k):
            heap.put(nums[i])
        for j in range(k, len(nums)):
            heap.put(nums[j])
            nums[index] = heap.get()
            index += 1
        while not heap.empty():
            nums[index] = heap.get()
            index += 1
        return nums

    def bucketSort(self, nums):
        # 桶排序中的基数排序
        def getDigit(num, d):
            # 返回当前数对应10的d次方位上的数字(即个十百……位)
            return num//(10**d) % 10

        bucket = [0]*10
        digit = 0
        for _ in nums:          # 获取数组中最大数的位数
            while _ // (10**digit):
                digit += 1
        for d in range(digit):  # 最大有几位数，就进行几次入桶出桶操作
            for num in nums:
                bucket[getDigit(num, d)] += 1
            for _ in range(1, 10):      # 每一位表示所有数中当前位小于等于该数的个数，即相当于出桶后该排的位置
                bucket[_] += bucket[_-1]
            temp = [0]*len(nums)
            for i in range(len(nums)-1, -1, -1):    # 从后往前遍历，即满足FIFO后入后出
                bucket[getDigit(nums[i], d)] -= 1
                temp[bucket[getDigit(nums[i], d)]] = nums[i]
            nums = temp
            bucket = [0]*10
        return nums

    def nodeDividedByValue(self, node, num):
        sh, st, eh, et, bh, bt = None, None, None, None, None, None     # 小于等于大于区域各自的头尾指针
        while node:
            if node.value < num:
                if st is not None:
                    st.next = node
                    st = node
                else:
                    sh, st = node, node
            elif node.value == num:
                if et is not None:
                    et.next = node
                    et = node
                else:
                    eh, et = node, node
            else:
                if bt is not None:
                    bt.next = node
                    bt = node
                else:
                    bh, bt = node, node
            node = node.next
        # 将所有尾部节点的next进行串联和重置，避免又指回原node链表中该节点的next
        if st is not None:
            st.next = eh if et is not None else bh
        if et is not None:
            et.next = bh
        bt.next = None      # 注释该行即变成有环链表
        return sh if st is not None else eh if et is not None else bh

    def firstLoopNode(self, node):
        # 有环链表(可无限.next下去)则返回入环第一个节点，无环则返回None
        if node is None or node.next is None or node.next.next is None:
            return
        f, s = node.next.next, node.next
        while f != s:
            if f.next is None or f.next.next is None:
                return
            f = f.next.next
            s = s.next
        f = node        # 当快慢指针相遇，快指针回到头部变慢重新出发，慢指针原位继续，则下次相遇即为入环第一个节点(证明略)
        while f != s:
            f = f.next
            s = s.next
        return f

    def noLoopFirstCrossNode(self, node1, node2):
        if node1 is None or node2 is None:
            return
        temp1, temp2, diff = node1, node2, 0
        while temp1.next:
            diff += 1
            temp1 = temp1.next
        while temp2.next:
            diff -= 1
            temp2 = temp2.next
        if temp1 != temp2:          # 两链表的最后一个节点不是同一个，则无相交
            return
        temp1 = node1 if diff > 0 else node2            # 绑定长链表
        temp2 = node2 if temp1 == node1 else node1
        diff = diff if diff > 0 else -diff
        while diff:                         # 长链表先走差值步，再一起走，相遇即第一个交叉节点
            temp1 = temp1.next
            diff -= 1
        while temp1 != temp2:
            temp1 = temp1.next
            temp2 = temp2.next
        return temp1

    def getFirstCrossNode(self, node1, node2):
        loop1 = self.firstLoopNode(node1)
        loop2 = self.firstLoopNode(node2)
        if loop1 == loop2:
            if loop1 is None:                               # 两无环链表求第一个交点
                return self.noLoopFirstCrossNode(node1, node2)
            else:                                           # 两链表相交且有公共环且相同点入环，以入环点为界限
                temp1, temp2, diff = node1, node2, 0
                while temp1.next != loop1:
                    diff += 1
                    temp1 = temp1.next
                while temp2.next != loop1:
                    diff -= 1
                    temp2 = temp2.next
                temp1 = node1 if diff > 0 else node2
                temp2 = node2 if temp1 == node1 else node1
                diff = diff if diff > 0 else -diff
                while diff:
                    temp1 = temp1.next
                    diff -= 1
                while temp1 != temp2:
                    temp1, temp2 = temp1.next, temp2.next
                return temp1
        elif loop1 is not None and loop2 is not None:
            temp1 = loop1.next
            while temp1 != loop1:
                if temp1 == loop2:                          # 两链表有公共环但在不同点入环，返回两入环点任一均可
                    return loop1
                temp1 = temp1.next
            return                                          # 两链表各自一个环，不相交
        else:                                               # 一有环一无环不相交
            return

    def treeOrderRecursion(self, tree):
        # 递归序打印二叉树
        if tree is None:
            return
        print(tree.value, end=' ')      # 前序遍历
        self.treeOrderRecursion(tree.left)
        # print(tree.value, end=' ')      # 中序遍历
        self.treeOrderRecursion(tree.right)
        # print(tree.value, end=' ')      # 后序遍历

    def treeOrderNonRecursion(self, tree, order):
        # 非递归打印二叉树
        def preOrder():
            # 前序 头节点入栈  弹出一个节点，若该节点有子节点，先入右再入左，重复直至空栈
            stack = [tree]
            while stack:
                cur = stack.pop()
                print(cur.value, end=' ')
                if cur.right:
                    stack.append(cur.right)
                if cur.left:
                    stack.append(cur.left)

        def inOrder():
            # 中序
            stack = []
            cur = tree
            while stack or cur:
                while cur:
                    stack.append(cur)
                    cur = cur.left
                cur = stack.pop()
                print(cur.value, end=' ')
                cur = cur.right

        def postOrder():
            # 后序 头节点入栈  弹出一个节点进入备用栈，若该节点有子节点，先压左再压右，重复直到所有节点进备用栈，最后依次弹出备用栈(左右头即头右左的逆序)
            stack1 = [tree]
            stack2 = []
            while stack1:
                cur = stack1.pop()
                stack2.append(cur)
                if cur.left:
                    stack1.append(cur.left)
                if cur.right:
                    stack1.append(cur.right)
            while stack2:
                print(stack2.pop().value, end=' ')

        if tree is None:
            return
        if order == 1:
            preOrder()
        if order == 2:
            inOrder()
        if order == 3:
            postOrder()

        # 深度优先遍历 即前序遍历
        # 宽度优先遍历


if __name__ == "__main__":
    def producer():
        temp = []
        for _ in range(1, 20):
            temp.append(random.randint(0, 10))
        return temp
    lmr = MyCollections()
    pro = producer()
    print(pro)
    head = Node(pro[0])
    temp = head
    for i in range(1, len(pro)):
        temp.next = Node(pro[i])
        temp = temp.next
    ttree = BinaryTree(1)
    ttree.left = BinaryTree(2)
    ttree.right = BinaryTree(3)
    ttree.left.left = BinaryTree(4)
    ttree.left.right = BinaryTree(5)
    ttree.right.left = BinaryTree(6)
    ttree.right.right = BinaryTree(7)
    # print(my == sorted(pro))
    lmr.treeOrderNonRecursion(ttree, 2)


