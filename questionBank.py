class Solution:
    # 1.数组中找出和为给定值的一组数并返回其下标
    def twoSum(self, nums, target):
        hashmap = dict()
        for i, val in enumerate(nums):
            if target - val in hashmap:
                return [hashmap[target - val], i]
            else:
                hashmap[val] = i

    # 2.反转输出一个带符号的32位整数，其范围为[-2**31, 2**31 - 1]，若溢出则返回0
    def reverse(self, num):
        flag = 1
        if num < 0:
            flag = - flag
            num = - num
        temp = int(str(num)[::-1]) * flag
        return temp if -2 ** 31 - 1 < temp < 2 ** 31 else 0

    # 3.判断一个整数是否为回文数，从左读和从右读是同一个数(>=0)
    def isPalindrome(self, num):
        return str(num)[::-1] == str(num)
        # (1)不转换为字符串
        # ans = 0
        # temp = num
        # if num == 0:
        #     return 1 == 1
        # if num < 0 or num % 10 == 0:
        #     return 1 == 0
        # else:
        #     while num > 0:
        #         ans *= 10
        #         ans += num % 10
        #         num //= 10
        #     if ans == temp:
        #         return 1 == 1
        #     else:
        #         return 1 == 0

    # 4.罗马数字转整数(IV=4, IX=9, XL=40, XC=90, CD=400, CM=900)
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


if __name__ == '__main__':
    print(Solution().twoSum(nums=[0, 2, 4, 3], target=5))
    print(Solution().isPalindrome(1234321))
    print(Solution().romanToInt('MCMXCIV'))
