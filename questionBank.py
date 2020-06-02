class Solution:
    # 1.数组中找出和为给定值的一组数并返回其下标
    def twoSum(self, nums, target):
        hashmap = dict()
        for i, val in enumerate(nums):
            if target - val in hashmap:
                return [hashmap[target - val], i]
            else:
                hashmap[val] = i


if __name__ == '__main__':
    print(Solution().twoSum(nums=[0, 2, 4, 3], target=5))
