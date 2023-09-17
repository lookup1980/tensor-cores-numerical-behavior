import os
import re

for f in os.listdir('.'):
    m = re.search('^test-(\S+)$', f)
    if m:
        result = 'result-' + m.group(1) + '.txt'
        print(result)
        os.system('./' + f + ' > ' + result)