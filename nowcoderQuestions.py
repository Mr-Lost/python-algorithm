class Nowcoder:
    def all_prime(self, num):
        # HJ6质数因子：分解成所有质数相乘
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


if __name__ == "__main__":
    lmr = Nowcoder()
    num = 8350
    lmr.all_prime(num)


