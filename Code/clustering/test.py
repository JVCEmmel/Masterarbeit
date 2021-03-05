import time

start = time.localtime()
time.sleep(10)
print(time.strftime("%H:%M:%S", start))
finish = time.localtime()
print(time.strftime("%H:%M:%S", finish))
# print(time.strftime("%H:%M:%S", finish) - time.strftime("%H:%M:%S", start))
duration = finish - start
print(time.strftime("%H:%M:%S", duration))