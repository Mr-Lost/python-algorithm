import re


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    """
    easy
    """
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

    # 69.x的平方根
    def mySqrt(self, x):
        if x == 1:
            return x
        left, right = 0, x
        while left < right-1:
            flag = (left+right)//2
            tem = flag ** 2
            if tem > x:
                right = flag
            else:
                left = flag
        return left

    # 167.两数之和：nums下标从1开始
    def twoSum2(self, nums, target):
        hashmap = dict()
        for i, val in enumerate(nums):
            if target - val in hashmap:
                return [hashmap[target - val]+1, i+1]
            else:
                hashmap[val] = i

    # 242.有效的字母异位词
    def isAnagram(self, s, t):
        for i in set(s):
            if s.count(i) != t.count(i):
                return False
        return len(s) == len(t)

    # 350.两个数组的交集
    def intersect(self, nums1, nums2):
        nums1.sort()
        nums2.sort()
        ans = []
        o, t = 0, 0
        while o < len(nums1) and t < len(nums2):
            if nums1[o] == nums2[t]:
                ans.append(nums1[o])
                o, t = o+1, t+1
            elif nums1[o] < nums2[t]:
                o += 1
            elif nums1[o] > nums2[t]:
                t += 1
        return ans

    # 482.密钥格式化
    def licenseKeyFormatting(self, s, k):
        ss = ''.join(s.upper().split('-'))[::-1]
        return '-'.join(ss[i:i+k] for i in range(0, len(ss), k))[::-1]

    # 605.种花问题：把尽量多个0变成1而使没有两个1挨在一起
    def canPlaceFlowers(self, flowerbed, n):
        ans = 0
        flowerbed = [0] + flowerbed + [0]
        for i, f in enumerate(flowerbed[1:-1]):
            if f == 0 and flowerbed[i] == 0 and flowerbed[i+2] == 0:
                ans += 1
                flowerbed[i+1] = 1
        return ans >= n

    # 628.三个数的最大乘积
    def maximumProduct(self, nums):
        nums.sort()
        return max(nums[0]*nums[1]*nums[-1], nums[-1]*nums[-2]*nums[-3])

    # 771.宝石与石头
    def numJewelsInStones(self, jewels, stones):
        return sum([stones.count(i) for i in jewels])

    # 804.唯一摩尔斯密码词：所有输入字符串中的不同翻译结果
    def uniqueMorseRepresentations(self, words):
        maps = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
        ans = set()
        for word in words:
            s = ''
            for w in word:
                s += maps[ord(w)-ord('a')]
            ans.add(s)
        return len(ans)

    # 922.按奇偶排序数组：索引与数值同奇偶性
    def sortArrayByParity(self, nums):
        ans = [0]*len(nums)
        nums.sort(key=lambda x: x % 2)   # 偶数余0排在前，奇数余1排在后
        ans[1::2], ans[::2] = nums[len(nums)//2:], nums[:len(nums)//2]
        return ans

    # 976.三角形的最大周长
    def largestPerimeter(self, nums):
        nums = sorted(nums, reverse=True)
        for i in range(len(nums)-2):
            if nums[i] < (nums[i+1]+nums[i+2]):
                return sum(nums[i:i+3])
        return 0

    # 1078.Bigram分词
    def findOcurrences(self, text, first, second):
        ans = []
        temp = text.split()
        for i in range(2, len(temp)):
            if temp[i-2] == first and temp[i-1] == second:
                ans.append(temp[i])
        return ans

    # 1207.独一无二的出现次数：所有数的出现次数都不同则返回true
    def uniqueOccurrences(self, arr):
        dic = dict()
        for i in arr:
            dic[i] = dic.get(i, 0) + 1
        tem = [dic[j] for j in dic]
        return len(set(tem)) == len(tem)

    # 1356.根据数字二进制下1的数目排序
    def sortByBits(self, arr):
        arr.sort()
        arr.sort(key=lambda x: bin(x).count('1'))
        return arr

    # 1436.旅行终点站
    def destCity(self, paths):
        city = dict()
        for s, _ in paths:
            city[s] = 1
        for _, d in paths:
            if not city.get(d, 0):
                return d

    # 1491.去掉最低工资和最高工资后的工资平均值
    def average(self, salary):
        return sum(sorted(salary)[1:-1]) / (len(salary)-2)

    # 1502.判断能否形成等差数列
    def canMakeArithmeticProgression(self, arr):
        # if len(arr) == 2:
        #     return True
        # arr.sort()
        # gap = arr[1] - arr[0]
        # for index, a in enumerate(arr[2:]):
        #     if a - arr[index+1] != gap:
        #         return False
        # return True
        arr.sort()
        tem = [arr[i+1]-arr[i] for i in range(len(arr)-1)]
        return len(set(tem)) == 1

    # 1512.好数对的数目：数字相同的组合数的总数
    def numIdenticalPairs(self, nums):
        return sum(nums[index+1:].count(i) for index, i in enumerate(nums))

    # 1576.替换所有的问号
    def modifyString(self, s):
        chars = 'abcdefghijklmnopqrstuvwxyz'
        s = '_' + s + '_'
        for i in range(1, len(s)-1):
            if s[i] == '?':
                for c in chars:
                    if c != s[i-1] and c != s[i+1]:
                        s = s[:i] + c + s[i+1:]
                        break
        return s[1:-1]

    """
    normal
    """
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

    # 11.盛最多水的容器：以数列中(index(i), i)在坐标系中做点，找出其中两点到x轴的垂线和x轴，三线围成的最大容器
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

    # 16.10面试题生存人：1900-2000中存活人数最多的一年
    def maxAliveYear(self, birth, death):
        area = [0]*101
        for i in range(len(birth)):
            for j in range(birth[i], death[i]+1):
                area[j-1900] += 1
        return area.index(max(area))+1900

    # 16.21面试题交换和
    def findSwapValues(self, array1, array2):
        gap = sum(array1) - sum(array2)
        if abs(gap) % 2:   # 差值为偶数才可能
            return []
        for i in set(array1):
            if i-gap//2 in set(array2):   # 集合去重避免超时
                return [i, i-gap//2]
        return []

    # 17.11面试题单词距离：指定两个单词出现的最近距离
    def findClosest(self, words, word1, word2):
        i, ans = 0, len(words)
        for j, word in enumerate(words):
            if word == word1 or word == word2:
                if word != words[i] and (words[i] == word1 or words[i] == word2):
                    ans = min(ans, j-i)
                i = j
        return ans

    # 56.合并区间：返回覆盖的所有区间段
    def merge(self, intervals):
        ans = []
        intervals.sort()
        left, right = intervals[0]
        for i in intervals[1:]:
            if i[0] > right:
                ans.append([left, right])
                left = i[0]
            right = max(right, i[1])
        ans.append([left, right])
        return ans

    # 442.数组中重复的数据：某些数重复出现了两次，其他都只有一次
    def findDuplicates(self, nums):
        ans = []
        for num in nums:
            k = abs(num)
            if nums[k-1] < 0:
                ans.append(k)
            nums[k-1] = -nums[k-1]   # 标记数字对应的索引的数值为负
        return ans

    # 468.验证IP地址
    def validIPAddress(self, ip):
        if '.' in ip:
            ips = ip.split('.')
            if len(ips) != 4:
                return "Neither"
            for ii in ips:
                try:
                    if ii.startswith('0') and len(ii) != 1:
                        return "Neither"
                    elif int(ii) < 0 or int(ii) > 255:
                        return "Neither"
                except:
                    return "Neither"
            return "IPv4"
        elif re.match(r"^(?:[a-f0-9]{1,4}:){7}[a-f0-9]{1,4}$", ip, re.I):
            return "IPv6"
        else:
            return "Neither"

    # 678.有效的括号字符串
    def checkValidString(self, s):
        star, left = [], []
        for i in range(len(s)):
            if s[i] == '(':
                left.append(i)
            if s[i] == '*':
                star.append(i)
            if s[i] == ')':
                if len(left) > 0:
                    left.pop()
                elif len(star) > 0:
                    star.pop()
                else:
                    return False
        while star and left:
            if star.pop() < left.pop():
                return False
        return len(left) <= len(star)

    # 739.每日温度：找出几天后升温
    def dailyTemperatures(self, temperatures):
        length = len(temperatures)
        ans = [0] * length
        stack = []
        for i in range(length):
            temperature = temperatures[i]
            while stack and temperature > temperatures[stack[-1]]:
                prev_index = stack.pop()
                ans[prev_index] = i - prev_index
            stack.append(i)
        return ans

    # 833.字符串中的查找与替换
    def findReplaceString(self, s, indexes, sources, targets):
        temp = sorted([[indexes[i], sources[i], targets[i]] for i in range(len(indexes))], reverse=True)
        for index, source, target in temp:
            if s[index:index+len(source)] == source:
                s = s[:index] + target + s[index+len(source):]
        return s

    # 904.水果成篮
    def totalFruit(self, fruits):
        pass

    # 1024.视频拼接
    def videoStitching(self, clips, t):
        dp = [101]*(t+1)
        dp[0] = 0
        for i in range(1, t+1):
            for ch in clips:
                if ch[0] < i <= ch[1]:
                    dp[i] = min(dp[i], dp[ch[0]]+1)
        return dp[t] if dp[t] != 101 else -1

    # 1054.距离相等的条形码：所有相邻数都不相同
    def rearrangeBarcodes(self, barcodes):
        length = len(barcodes)
        counts = dict()
        for barcode in barcodes:
            counts[barcode] = counts.get(barcode, 0)+1
        bars = sorted([(-val, key) for key, val in counts.items()])  # 负号相当于从多到少排序
        temp = []
        for val, key in bars:
            temp += [key]*(-val)
        ans = [0]*length
        j = 0
        for i in range(0, length, 2):
            ans[i] = temp[j]
            j += 1
        for k in range(1, length, 2):
            ans[k] = temp[j]
            j += 1
        return ans

    # 1222.可以攻击国王的皇后
    def queenAttacktheKing(self, queens, king):
        board = [[0 for _ in range(8)] for _ in range(8)]
        ans = []
        for i, j in queens:
            board[i][j] = 1
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        for dix, diy in directions:
            x, y = king[0] + dix, king[1] + diy
            while 0 <= x < 8 and 0 <= y < 8:
                if board[x][y] == 1:
                    ans.append([x, y])
                    break
                x, y = x + dix, y + diy
        return ans

    # 1333.餐厅过滤器
    def filterRestaurants(self, restaurants, veganFriendly, maxPrice, maxDistance):
        restaurants.sort()
        ans = []
        for index, rest in enumerate(restaurants):
            if veganFriendly == (veganFriendly & rest[2]) and rest[3] <= maxPrice and rest[4] <= maxDistance:
                ans.append(index)
        ans.sort(key=lambda x: restaurants[x][1])
        for index, i in enumerate(ans):
            ans[index] = restaurants[i][0]
        return ans[::-1]

    # 1451.重新排列句子中的单词
    def arrangeWords(self, text):
        text = text.lower().split()
        text.sort(key=lambda x: len(x))
        return ' '.join(text).capitalize()

    # 1452.收藏清单
    def peapleIndexes(self, favoriteCompanies):
        ans = []
        favoriteCompanies = [set(fc) for fc in favoriteCompanies]
        for index, fC in enumerate(favoriteCompanies):
            for i in range(len(favoriteCompanies)):
                if fC < favoriteCompanies[i]:
                    break
                if i == len(favoriteCompanies) - 1:
                    ans.append(index)
        return ans

    # 1456.定长子串中元音的最大数目
    def maxVowels(self, s, k):
        vowels = 'aeiou'
        temp, left = 0, 0
        for i in range(k):
            if s[i] in vowels:
                temp += 1
        ans = temp
        for i in range(k, len(s)):
            if s[i] in vowels:
                temp += 1
            if s[left] in vowels:
                temp -= 1
            ans = max(ans, temp)
            left += 1
        return ans

    # 1481.不同整数的最少数目
    def findLeastNumOfUniqueInts(self, arr, k):
        # nums = sorted([arr.count(i) for i in set(arr)])   # 可能超时
        nums = {}
        for i in arr:
            nums[i] = nums.get(i, 0) + 1
        nums = sorted(list(nums.values()))
        for index, num in enumerate(nums):
            if k - num >= 0:
                k -= num
            else:
                return len(nums[index:])
        return 0

    # 1529.灯泡开关
    def minFlips(self, target):
        return int(target[0]) + target.count('01') + target.count('10')

    # 1561.你可以获得的最大硬币数目
    def maxCoins(self, piles):
        piles = sorted(piles, reverse=True)
        return sum(piles[:len(piles)*2//3+1][1::2])

    # 1647.字符频次唯一的最小删除次数
    def minDeletions(self, s):
        temp = []
        ans = 0
        for i in set(s):
            temp.append(s.count(i))
        temp.sort(reverse=True)
        if len(temp) > 1:
            for t in range(1, len(temp)):
                while temp[t] > 0 and len(set(temp[:t+1])) < len(temp[:t+1]):
                    temp[t] -= 1
                    ans += 1
        return ans

    """
    difficult
    """
    # 1096.花括号展开


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
    # print(lmr.threeSum([0, 0, 0]))
    ss = [["leetcode","google","facebook"],["google","microsoft"],["google","facebook"],["google"],["amazon"]]
    print(lmr.peapleIndexes(ss))



