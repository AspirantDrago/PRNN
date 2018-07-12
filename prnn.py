import random
import hashlib
import math


class Neuron:
    def __init__(self, net, index, activ_func, alfa=1, is_input=False):
        self.net = net
        self.is_input = is_input
        self.n = net.mass
        self.x = [1] + [0] * self.n
        self.b = [0] * (self.n + 1)
        self.b_step = [0] * (self.n + 1)
        self.w = [0] * (self.n + 1)
        self.out = 0
        self.step = 0
        self.index = index
        self.arr = []
        self.init_arr()
        self.init_b()
        self.activ_func = activ_func
        self.alfa = alfa

    def init_arr(self):
        self.arr = []
        for i in range(self.n):
            catch1 = catch2 = False
            iter = index = 0
            while not (catch1 and catch2):
                iter += 1
                index = self.net.gethash(i + self.n * self.index, iter) % self.net.count
                catch1 = self.net.is_loop or (index != self.index)
                catch2 = self.net.is_duplicate or (self.arr.count(index) == 0)
            self.arr += [index]

    def activ(self, x):
        method = getattr(self.net, self.activ_func)
        return method(x, self.alfa)

    def summator(self, xs):
        s = 0
        for i in range(self.n + 1):
            self.w[i] = xs[i] * self.b[i]
            s += self.w[i]
        return s

    def summator_prime(self):
        s = self.summator(self.x)
        method = getattr(self.net, self.activ_func + '_prime')
        return method(s, self.alfa)

    def calcing(self, step=0):
        if (not self.is_input) and (step != self.step):
            self.step = step
            for i in range(1, 1 + self.n):
                self.net.arr[self.arr[i - 1]].calcing(step)
                self.x[i] = self.net.arr[self.arr[i - 1]].out
            self.out = self.activ(self.summator(self.x))
        return self.out

    def calc(self):
        step = random.random()
        r1 = self.calcing(step)
        if self.net.is_stabilization:
            step = random.random()
            r2 = self.calcing(step)
            while abs(r1 - r2) > self.net.delta:
                # print('Stabilization:', abs(r1 - r2))
                r1 = r2
                step = random.random()
                r2 = self.calcing(step)

    def init_b(self):
        for i in range(self.n + 1):
            #w = random.normalvariate(0, self.net.rand_sigma)
            w = random.uniform(- self.net.rand_sigma, self.net.rand_sigma)
            while abs(w) < self.net.rand_delta:
                #w = random.normalvariate(0, self.net.rand_sigma)
                w = random.uniform(- self.net.rand_sigma, self.net.rand_sigma)
            self.b[i] = w
        return self.b

    def setvalue(self, value):
        self.out = value

    def getregular(self):
        if self.net.regular_level <= 0:
            return 0
        s = 0
        for i in range(1, self.n + 1):
            s += self.net.regular_coeff * (abs(self.b[i]) ** self.net.regular_level) / self.net.regular_level
        return s

    def getdentr(self, step, sorting=True):
        l = zip(self.x, self.b_step, list(range(len(self.w))))
        l = filter(lambda x: x[1] != step, l)
        if sorting:
            l = sorted(l, key=lambda x: abs(x[0]), reverse=True)
        return list(l)

    def learning1(self, err, step=0):
        if not self.is_input:
            err *= self.summator_prime()
            err += self.getregular()
            dentr = self.getdentr(step)
            if len(dentr) > 0:
                n = dentr[0][2]
                if n >= 0:
                    self.b_step[n] = step
                    self.b[n] += self.net.speed * err * self.x[n]
                    if n > 0:
                        err *= self.b[n]
                        self.net.arr[self.arr[n - 1]].learning1(err, step)

    def learn1(self, err):
        step = random.random()
        for i in range(self.net.count_learn):
            self.learning1(err, step)

    def learning2(self, err, step = 0):
        if not self.is_input:
            err *= self.summator_prime()
            err += self.getregular()
            dentr = self.getdentr(step, False)
            for i in range(len(dentr)):
                n = dentr[i][2]
                if n >= 0:
                    self.b_step[n] = step
                    self.b[n] += self.net.speed * err * self.x[n]
                    if n > 0:
                        err *= self.b[n]
                        self.net.arr[self.arr[n - 1]].learning2(err, step)

    def learn2(self, err):
        step = random.random()
        self.learning2(err, step)

    def learning3(self, err, step = 0):
        if not self.is_input:
            err *= self.summator_prime()
            err += self.getregular()
            dentr = self.getdentr(step, False)
            l = len(dentr)
            l_err = [err] * l
            for i in range(l):
                n = dentr[i][2]
                if n >= 0:
                    self.b_step[n] = step
                    l_err[i] *= self.b[n]
                    self.b[n] += self.net.speed * err * self.x[n]
            for i in range(l):
                n = dentr[i][2]
                if n > 0:
                    self.net.arr[self.arr[n - 1]].learning3(l_err[i], step)

    def learn3(self, err):
        step = random.random()
        self.learning3(err, step)

class NeuroNet:
    # Инициализация нейросети
    def __init__(self, count_in, count_out, count_n=0, mass=2):
        self.count_in = 0                           # Число входных нейронов
        self.count_out = 0                          # Число выходных нейронов
        self.count_n = 0                            # Число скрытых нейронов
        self.count = 0                              # Общее число нейронов
        self.key = 1                                # Ключ
        self.min_key = 1                            # Минимальная граница ключа
        self.max_key = 10 ** 9                      # Максимальная граница ключа
        self.activ_func = 'logistic'                # Активационная функция
        self.rand_sigma = 0.5                       # Среднеквадратичное отклонение нормального распределения для весов
        self.rand_delta = .1 ** 6                  # Минимальное значение веса по модулю
        self.alfa = 1.0                             # Крутизна активационных функций
        self.mass = 0                               # Число дентритных связей
        self.speed = 0.1                            # Скорость обучения
        self.arr = []                               # Список нейронов
        self.norm_in = []                           # Массив кортежей нормировки входных значений
        self.norm_out = []                          # Массив кортежей нормировки выходных значений
        self.disc_out = []                          # Массив дискретности выходных значений
        self.delta = 0.001                          # Коэффициент сходимости
        self.count_learn = 0                        # Кратность обучения выходного нейрона
        self.is_duplicate = False                   # Наличие дублирующихся дентритных связей
        self.is_loop = False                        # Наличие единичных петель
        self.regular_level = 0                      # Уровень регулиризации весов
        self.regular_coeff = 1                      # Коэффициент регуляризации весов
        self.is_stabilization = False               # Требуется ли стабилизация
        self.error_degree = 2                       # Степень в функционале ошибки
        self.getkey()
        self.frame(count_in, count_out, count_n, mass)

    # Задание структуры
    def frame(self, count_in, count_out, count_n=0, mass=2):
        self.count_in = max(int(count_in), 1)
        self.count_out = max(int(count_out), 1)
        self.count_n = max(int(count_n), 0)
        self.count = self.count_in + self.count_out + self.count_n
        self.mass = min(max(int(mass), 1), self.count + 1)
        self.count_learn = self.count_in + 1
        self.norm_in = [(-1, 1, -1, 1)] * self.count_in
        if self.activ_func == 'tanh' or self.activ_func == 'sinh' or self.activ_func == 'identity':
            self.norm_out = [(-1, 1, -1, 1)] * self.count_out
        else:
            self.norm_out = [(0, 1, 0, 1)] * self.count_out
        self.disc_out = [False] * self.count_out
        self.reinit()

    def reinit(self):
        for i in range(self.count_in):
            self.arr.append(Neuron(self, i, self.activ_func, self.alfa, True))
        for i in range(self.count_out + self.count_n):
            self.arr.append(Neuron(self, i + self.count_in, self.activ_func, self.alfa))

    def logistic(self, x, alfa):
        return 1 / (1 + math.exp(- alfa * x))

    def logistic_prime(self, x, alfa):
        return self.logistic(x, alfa) * (1 - self.logistic(x, alfa))

    def exponential(self, x, alfa):
        return math.exp(alfa * x)

    def exponential_prime(self, x, alfa):
        return alfa * math.exp(alfa * x)

    def tanh(self, x, alfa):
        return math.tanh(alfa * x)

    def tanh_prime(self, x, alfa):
        return alfa / (math.cosh(alfa * x) ** 2)

    def sinh(self, x, alfa):
        return math.sinh(alfa * x)

    def sinh_prime(self, x, alfa):
        return alfa * math.cosh(alfa * x)

    def identity(self, x, alfa):
        return alfa * x

    def identity_prime(self, x, alfa):
        return alfa

    def relu(self, x, alfa):
        return max(alfa * x, 0)

    def relu_prime(self, x, alfa):
        return max(alfa, 0)

    def hlim(self, x, alfa):
        if x >= 0:
            return 1
        return 0

    def hlim_prime(self, x, alfa):
        return 0

    def module(self, x, alfa):
        return abs(alfa * x)

    def module_prime(self, x, alfa):
        if x >= 0:
            return alfa
        return - alfa

    def seterrordegree(self, degree):
        self.error_degree = int(degree)

    def setregular(self, level=0, coeff=1):
        self.regular_level = abs(int(level))
        self.regular_coeff = float(coeff)

    def setdiscretout(self, vector):
        self.disc_out = vector

    def setcountlearn(self, count):
        self.count_learn = abs(int(count))

    def setalfa(self, alfa):
        self.alfa = alfa

    def normal(self, value, norm, revers=False):
        min1 = min(norm[0], norm[1])
        max1 = max(norm[0], norm[1])
        min2 = min(norm[2], norm[3])
        max2 = max(norm[2], norm[3])
        if revers:
            min1, max1, min2, max2 = min2, max2, min1, max1
        value = (value - min1) * (max2 - min2) / float(max1 - min1) + min2
        if value > max2:
            value = max2
        if value < min2:
            value = min2
        return value

    def setinnormal(self, vector):
        self.norm_in = vector

    def setoutnormal(self, vector):
        self.norm_out = vector

    def setkey(self, value):
        self.key = int(value)

    def setdelta(self, delta):
        self.delta = delta

    def getkey(self):
        value = random.randint(self.min_key, self.max_key)
        self.setkey(value)
        return value

    def gethash(self, value, iter=1):
        value += self.key * iter
        r = hashlib.md5(str(value).encode('utf-8')).hexdigest()
        return int(r, 16)

    def gety(self, row):
        row = list(row)
        if len(row) < self.count_in:
            row += [0] * (self.count_in - len(row))
        else:
            row = row[:self.count_in]
        for i in range(self.count_in):
            self.arr[i].setvalue(self.normal(row[i], self.norm_in[i]))
        for neur in self.arr[self.count_in: self.count_in + self.count_out]:
            neur.calc()
        y = []
        for i in range(self.count_in, self.count_in + self.count_out):
            val = self.normal(self.arr[i].out, self.norm_out[i - self.count_in], True)
            if self.disc_out[i - self.count_in]:
                val = round(val)
            y += [val]
        return y

    def geterror(self, row):
        row = list(row)
        rez = row[- self.count_out:]
        rezn = self.gety(row[: - self.count_out])
        err = 0
        for i in range(len(rez)):
            err += abs(rez[i] - rezn[i]) ** self.error_degree
        err /= self.error_degree
        return err, rezn

    def setspeed(self, speed):
        self.speed = speed

    def learn(self, row):
        row = list(row)
        rez = row[- self.count_out:]
        rezn = self.gety(row[: - self.count_out])
        err = 0
        for i in range(len(rez)):
            coeff = abs((self.norm_out[i][3] - self.norm_out[i][2]) / (self.norm_out[i][1] - self.norm_out[i][0]))
            val = (coeff * abs(rez[i] - rezn[i])) ** self.error_degree
            val /= self.error_degree
            err += val
            if rez[i] < rezn[i]:
                val *= -1
            rez[i] = val
        for i in range(len(rez)):
            self.arr[i + self.count_in].learn3(rez[i])
        return err, rezn