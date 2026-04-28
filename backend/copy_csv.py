import shutil, os
src = r'D:\supply\backend\delhi_100_orders_unique.csv'
dst = r'D:\supply\backend\delhi_100_orders.csv'
if os.path.exists(src):
    shutil.copy(src, dst)
    print('COPIED', dst)
else:
    print('SRC NOT FOUND', src)
