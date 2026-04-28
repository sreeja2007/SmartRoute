import requests, json, os
url = 'http://127.0.0.1:5000/upload-csv'
path = os.path.join(os.path.dirname(__file__), 'delhi_100_orders.csv')
print('Uploading', path)
with open(path, 'rb') as f:
    files = {'file': ('delhi_100_orders.csv', f, 'text/csv')}
    r = requests.post(url, files=files, timeout=120)
    print('UPLOAD STATUS', r.status_code)
    print(r.text[:1000])

r2 = requests.post('http://127.0.0.1:5000/api/reset_simulation', json={}, timeout=120)
print('RESET:', r2.status_code, r2.text)

r3 = requests.get('http://127.0.0.1:5000/api/get_simulation_status', timeout=120)
js = r3.json()
print('ORDERS COUNT:', len(js['simulation']['orders']))
print('SAMPLE 5 ORDERS:')
for o in js['simulation']['orders'][:5]:
    print(o)
