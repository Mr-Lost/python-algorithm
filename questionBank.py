import re


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # #easy
    # 1.两数之和：数组中找出和为给定值的一组数并返回其下标
    def twoSum(self, nums, target):
        hashmap = dict()
        for i, val in enumerate(nums):
            if target - val in hashmap:
                return [hashmap[target - val], i]
            else:
                hashmap[val] = i

    # 7.整数反转：反转输出一个带符号的32位整数，其范围为[-2**31, 2**31 - 1]，若溢出则返回0
    def reverse(self, num):
        flag = 1
        if num < 0:
            flag = - flag
            num = - num
        temp = int(str(num)[::-1]) * flag
        return temp if -2 ** 31 - 1 < temp < 2 ** 31 else 0

    # 9.回文数：判断一个整数是否为回文数，从左读和从右读是同一个数(>=0)
    def isPalindrome(self, num):
        return str(num)[::-1] == str(num)
        # (1)不转换为字符串
        # ans = 0
        # temp = num
        # if num == 0:
        #     return True
        # if num < 0 or num % 10 == 0:
        #     return False
        # else:
        #     while num > 0:
        #         ans *= 10
        #         ans += num % 10
        #         num //= 10
        #     if ans == temp:
        #         return True
        #     else:
        #         return False

    # 13.罗马数字转整数(IV=4, IX=9, XL=40, XC=90, CD=400, CM=900)
    def romanToInt(self, roman):
        temp = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        ans = 0
        for i in range(len(roman)):
            if i < len(roman)-1 and temp.get(roman[i]) < temp.get(roman[i+1]):
                ans -= temp.get(roman[i])
            else:
                ans += temp.get(roman[i])
        return ans
        # (1)这样定义双字母值，便于下面循环操作中值的叠加
        # temp = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000,
        #         'IV': 3, 'IX': 8, 'XL': 30, 'XC': 80, 'CD': 300, 'CM': 800}
        # ans = 0
        # for i in range(len(roman)):
        #     if roman[i-1:i+1] in temp:   # roman[-1:1]=''
        #         ans += temp.get(roman[i-1:i+1])
        #     else:
        #         ans += temp.get(roman[i])
        # return ans
        # (2)
        # temp = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000,
        #         'IV': 3, 'IX': 8, 'XL': 30, 'XC': 80, 'CD': 300, 'CM': 800}
        # return sum(temp.get(roman[i-1:i+1], temp[n]) for i, n in enumerate(roman))

    # 14.最长公共前缀：查找字符串数组中最长公共前缀
    def longestCommonPrefix(self, strs):
        ans = ''
        for i in zip(*strs):
            if len(set(i)) == 1:
                ans += i[0]
            else:
                break
        return ans

    # 20.有效的括号：判断左右括号闭合是否有效
    def isValid(self, s):
        temp = {'(': ')', '[': ']', '{': '}', '/': '/'}
        stack = ['/']   # 赋初值为了避免没有元素时，pop()报错
        for i in s:
            if i in temp:
                stack.append(temp[i])
            elif stack.pop() != i:
                return False
        return len(stack) == 1

    # 26.删除排序数组中的重复项：删除一个排序数列中重复的元素后，返回新数组的长度
    def removeDuplicates(self, nums):
        nums.sort()
        for i in range(len(nums)-1, 0, -1):
            if nums[i] == nums[i-1]:
                nums.pop(i)
        return len(nums)

    # 27.移除元素：删除数组中所有的某个特定值，并返回数组长度
    def removeElement(self, nums, val):
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] == val:
                nums.pop(i)
        return len(nums)
        # (1)不删除元素，仅返回数组长度
        # return len(nums) - nums.count(val)

    # 28.在字符串1中找到字符串2出现的第一个位置
    def strStr(self, str1, str2):
        if str2 == '':
            return 0
        for index, i in enumerate(str1):
            if i == str2[0]:
                if str1[index:index+len(str2)] == str2:
                    return index
        return -1
        # (1)
        # return str1.find(str2) if str2 != '' else 0

    # 35.搜索插入位置：在一个升序排列的数组中查找指定元素，有则返回下标，无则返回按序插入的位置
    def searchInsert(self, nums, target):
        if target not in nums:
            nums.append(target)
            nums.sort()
        return nums.index(target)
        # (1)二分法
        # if target > nums[-1]:
        #     return len(nums)
        # mid = len(nums) // 2
        # temp = nums[:mid+1] if target < nums[mid] else nums[mid:]
        # for i in temp:
        #     if target <= i:
        #         return nums.index(i)

    # 38.外观数列：下一项描述上一项数值依次是a个x，b个y，c个z ... (xyyxzzz -> 1x2y1x3z)
    def countAndSay(self, n):
        exam = '1'
        for i in range(n - 1):
            ans = ''
            count = 0
            current = exam[0]
            for index, j in enumerate(exam):
                if current == j:
                    count += 1
                else:
                    ans += '{}{}'.format(count, current)
                    count = 1
                    current = exam[index]
            ans += '{}{}'.format(count, current)
            exam = ans
        return exam
        # (1)递归
        # if n <= 1:
        #     return '1'
        # exam = self.countAndSay(n-1)
        # ans = ''
        # count = 1
        # for i in range(1, len(exam)):
        #     if exam[i-1] == exam[i]:
        #         count += 1
        #     else:
        #         ans += '{}{}'.format(count, exam[i-1])
        #         count = 1
        # ans += '{}{}'.format(count, exam[-1])
        # return ans

    # #normal
    # 2.两数相加：非空链表逆序存储两个正数，返回相加结果的新链表
    def addTwoNumbers(self, li1, li2):
        head = temp = ListNode(None)
        ans = 0
        while li1 or li2 or ans:
            ans += (li1.val if li1 else 0) + (li2.val if li2 else 0)
            temp.next = ListNode(ans % 10)
            temp = temp.next
            ans //= 10
            li1 = li1.next if li1 else None
            li2 = li2.next if li2 else None
        return head.next

    # 3.无重复字符的最长子串：返回字符串中最长不重复子串的长度
    def lengthOfLongestSubstring(self, s):
        temp = dict()
        index, ans = 0, 0
        for i in range(len(s)):
            if s[i] in temp:
                index = max(temp[s[i]], index)
            ans = max(ans, i - index + 1)
            temp[s[i]] = i + 1
        return ans

    # 5.最长回文子串
    def longestPalindrome(self, s):
        if len(s) < 2:
            return s
        ans = s[0]
        for i in range(len(s) - 1):
            for j in range(i, len(s)):
                temp = s[i:j + 1]
                if temp == temp[::-1] and len(temp) > len(ans):
                    ans = temp
        return ans
        # (1)超时
        # ans = ''
        # if len(s) < 2:
        #     return s
        # for i in range(len(s) - 1):
        #     for j in range(i + 1, len(s)):
        #         mid = (j + i) // 2
        #         temp = s[mid + 1:j + 1] if len(s[i:j+1]) % 2 == 0 else s[mid:j + 1]
        #         if s[i:mid + 1] == temp[::-1] and len(s[i:j + 1]) > len(ans):
        #             ans = s[i:j + 1]
        # return ans

    # 6.Z字形变换：将字符串各个字符从上到下再往上，往返逐行各插入一个字符，最后输出按行拼接的结果
    def convert(self, s, numRows):
        if numRows < 2:
            return s if numRows == 1 else ''
        temp = [''] * numRows
        row, flag = 0, 1
        for i in s:
            temp[row] += i
            row += flag
            if row == numRows - 1 or row == 0:
                flag = -flag
        return ''.join(temp)

    # 8.字符串转换整数：从第一个非空字符开始，第一个字符可以是正负号，然后连接尽可能多的数字，有则返回
    def myAtoi(self, s):
        temp = ''
        flag = 0   # 是否带正负号
        str1 = s.strip() + 'end'
        if str1[0] == '-' or str1[0] == '+':
            flag += 1
            temp += str1[0]
            str1 = str1[1:]
        for i in range(len(str1)):
            if str1[i].isdigit():
                temp += str1[i]
            else:
                break
        ans = 0 if len(temp) == flag else int(temp)   # ans=0当temp没有内容len(temp)=0，或仅有一个正负号的时候len(temp)=1
        return max(min(ans, 2 ** 31 - 1), -2 ** 31)
        # (1)正则
        # return max(min(int(*re.findall(r'^[-+]?\d+', s.lstrip())), 2**31 - 1), -2**31)

    # 9.盛最多水的容器：以数列中(index(i), i)在坐标系中做点，找出其中两点到x轴的垂线和x轴，三线围成的最大容器
    def maxArea(self, height):
        left, right = 0, len(height) - 1
        ans = 0
        while right > left:
            width = right - left
            temp = width * min(height[left], height[right])
            ans = temp if temp > ans else ans
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return ans

    # 12.整数转罗马数字：设定该数为1-3999
    def intToRoman(self, num):
        N = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
        n = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        ans = ''
        for index, i in enumerate(n):
            if num // i:
                ans += N[index] * (num // i)
                num -= i * (num // i)
        return ans

    # 15.三数之和
    def threeSum(self, nums):
        if len(nums) < 3:
            return []
        nums.sort()
        ans = []
        for i in range(len(nums)-1):
            if nums[i] > 0:   # 三个正数相加必不为0
                break
            if i > 0 and nums[i] == nums[i-1]:   # 过滤相同项
                continue
            left, right = i + 1, len(nums) - 1
            while left < right:
                temp = nums[i] + nums[left] + nums[right]
                if temp == 0:
                    ans.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    left, right = left + 1, right - 1
                elif temp > 0:
                    right -= 1   # 在第一个数不变的情况下，减小较大数来减小三数之和
                else:
                    left += 1   # 在第一个数不变的情况下，增大较小数来增大三数之和
        return ans


if __name__ == '__main__':
    lmr = Solution()
    # print(lmr.twoSum(nums=[0, 2, 4, 3], target=5))
    # print(lmr.isPalindrome(1234321))
    # print(lmr.romanToInt('MCMXCIV'))
    # print(lmr.longestCommonPrefix(['fly', 'float', 'flow']))
    # print(lmr.isValid('{[[([])]()]}'))
    # print(lmr.removeElement([1,2,3,3,2,4,5], 3))
    # print(lmr.strStr('good', ''))
    # print(lmr.countAndSay(4))
    # print(lmr.lengthOfLongestSubstring('acgbsdseiasdg'))
    # print(lmr.longestPalindrome('bb'))
    # print(lmr.convert('LEETCODEISHIRING', 3) == 'LCIRETOESIIGEDHN')
    # print(lmr.myAtoi('2.325'))
    print(lmr.threeSum([0,0,0]))


