"""Execute a notebook in place with jupyter_client (without nbconvert or nbclient).

Usage: python experiments/run_notebook.py notebooks/analysis.ipynb notebooks
"""
import json, sys, queue
from jupyter_client.manager import start_new_kernel

path, workdir = sys.argv[1], sys.argv[2]
nb = json.load(open(path))
km, kc = start_new_kernel(kernel_name='python3', cwd=workdir)
kc.execute_interactive("%matplotlib inline", timeout=60)
count = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    count += 1
    outputs = []
    def hook(msg):
        t, c = msg['header']['msg_type'], msg['content']
        if t == 'stream':
            if outputs and outputs[-1]['output_type'] == 'stream' and outputs[-1]['name'] == c['name']:
                outputs[-1]['text'] += c['text']
            else:
                outputs.append({'output_type': 'stream', 'name': c['name'], 'text': c['text']})
        elif t in ('display_data', 'execute_result'):
            o = {'output_type': t, 'data': c['data'], 'metadata': c.get('metadata', {})}
            if t == 'execute_result':
                o['execution_count'] = count
            outputs.append(o)
        elif t == 'error':
            outputs.append({'output_type': 'error', 'ename': c['ename'], 'evalue': c['evalue'], 'traceback': c['traceback']})
    reply = kc.execute_interactive(''.join(cell['source']), timeout=3600, output_hook=hook)
    for o in outputs:
        if o['output_type'] == 'stream':
            o['text'] = o['text'].splitlines(keepends=True)
    cell['outputs'], cell['execution_count'] = outputs, count
    status = reply['content']['status']
    print(f'cell {count}: {status}', (outputs[-1].get('ename', '') + ' ' + outputs[-1].get('evalue', '')[:100]) if status == 'error' else '')
kc.stop_channels(); km.shutdown_kernel()
json.dump(nb, open(path, 'w'), indent=1)
