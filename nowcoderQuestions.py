class Nowcoder:
    # HJ6质数因子：分解成所有质数相乘
    def all_prime(self, num):
        is_prime = True
        for i in range(2, int(num**0.5)+1):
            if num % i == 0:
                is_prime = False
                num //= i
                print(i, end=' x ')
                self.all_prime(num)
                break
        if is_prime:
            print(num)

    # NC17最长回文子串
    def longest_palindrome(self, word):
        for i in range(len(word)):
            for j in range(len(word)//2+1):
                pass

    # NC52有效括号序列
    def valid_brackets(self, word):
        stack = []
        dic = {'(': ')', '[': ']', '{': '}'}
        for i in word:
            if i in dic:
                stack.append(dic[i])
            else:
                if len(stack):
                    if i == stack.pop():
                        continue
                return False
        return not len(stack)

    # NC109岛屿数量
    def islands_num(self, grid):
        def dfs(grid, i, j):
            if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == '0':
                return
            grid[i][j] = '0'
            dfs(grid, i+1, j)
            dfs(grid, i-1, j)
            dfs(grid, i, j+1)
            dfs(grid, i, j-1)

        ans = 0
        if len(grid):
            if len(grid[0]):
                m, n = len(grid), len(grid[0])
                for x in range(m):
                    for y in range(n):
                        if grid[x][y] == '1':
                            ans += 1
                            dfs(grid, x, y)
        return ans


if __name__ == "__main__":
    lmr = Nowcoder()
    num = 8350
    lmr.all_prime(num)


