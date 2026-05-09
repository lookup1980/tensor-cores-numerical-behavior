import os
import re
import subprocess

for f in sorted(os.listdir('.')):
    m = re.search(r'^test-(\S+)$', f)
    if m:
        result = 'result-' + m.group(1) + '.txt'
        print(result)
        with open(result, 'w') as outfile:
            subprocess.run(['./' + f], stdout=outfile, check=True)
