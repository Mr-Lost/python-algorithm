class Solution:
    # #easy
    # 1.数组中找出和为给定值的一组数并返回其下标
    def twoSum(self, nums, target):
        hashmap = dict()
        for i, val in enumerate(nums):
            if target - val in hashmap:
                return [hashmap[target - val], i]
            else:
                hashmap[val] = i

    # 7.反转输出一个带符号的32位整数，其范围为[-2**31, 2**31 - 1]，若溢出则返回0
    def reverse(self, num):
        flag = 1
        if num < 0:
            flag = - flag
            num = - num
        temp = int(str(num)[::-1]) * flag
        return temp if -2 ** 31 - 1 < temp < 2 ** 31 else 0

    # 9.判断一个整数是否为回文数，从左读和从右读是同一个数(>=0)
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

    # 14.查找字符串数组中最长公共前缀
    def longestCommonPrefix(self, strs):
        ans = ''
        for i in zip(*strs):
            if len(set(i)) == 1:
                ans += i[0]
            else:
                break
        return ans

    # 20.判断左右括号闭合是否有效
    def isValid(self, s):
        temp = {'(': ')', '[': ']', '{': '}', '/': '/'}
        stack = ['/']   # 赋初值为了避免没有元素时，pop()报错
        for i in s:
            if i in temp:
                stack.append(temp[i])
            elif stack.pop() != i:
                return False
        return len(stack) == 1

    # 26.删除一个排序数列中重复的元素后，返回新数组的长度
    def removeDuplicates(self, nums):
        nums.sort()
        for i in range(len(nums)-1, 0, -1):
            if nums[i] == nums[i-1]:
                nums.pop(i)
        return len(nums)

    # 27.删除数组中所有的某个特定值，并返回数组长度
    def removeElement(self, nums, val):
        for i in range(len(nums) - 1, -1, -1):
            if nums[i] == val:
                nums.pop(i)
        return len(nums)
        # (1)不删除元素，仅返回数组长度
        # return len(nums) - nums.count(val)

    # 28. 在字符串1中找到字符串2出现的第一个位置
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

    # 35. 在一个升序排列的数组中查找指定元素，有则返回下标，无则返回按序插入的位置
    def searchInsert(self, nums, target):
        if target not in nums:
            nums.append(target)
            nums.sort()
        return nums.index(target)
        # (1) 二分法
        # if target > nums[-1]:
        #     return len(nums)
        # mid = len(nums) // 2
        # temp = nums[:mid+1] if target < nums[mid] else nums[mid:]
        # for i in temp:
        #     if target <= i:
        #         return nums.index(i)

    # 38.外观数列，下一项描述上一项数值依次是a个x，b个y，c个z ... (xyyxzzz -> 1x2y1x3z)
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


if __name__ == '__main__':
    lmr = Solution()
    # print(lmr.twoSum(nums=[0, 2, 4, 3], target=5))
    # print(lmr.isPalindrome(1234321))
    # print(lmr.romanToInt('MCMXCIV'))
    # print(lmr.longestCommonPrefix(['fly', 'float', 'flow']))
    # print(lmr.isValid('{[[([])]()]}'))
    # print(lmr.removeElement([1,2,3,3,2,4,5], 3))
    print(lmr.strStr('good', ''))


