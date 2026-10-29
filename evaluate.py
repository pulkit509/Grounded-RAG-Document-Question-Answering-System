"""Evaluate retrieval recall using labeled source/page questions."""
import argparse
import json
from src.core import Store

p = argparse.ArgumentParser()
p.add_argument('dataset', help='JSONL: question, source, page')
p.add_argument('--k', type=int, default=5)
args = p.parse_args()
store = Store()
rows = [json.loads(line) for line in open(args.dataset, encoding='utf-8') if line.strip()]
if not rows:
    raise SystemExit('Empty evaluation set')
correct = 0
for item in rows:
    hits = store.search(item['question'], args.k)
    found = any(h['source'] == item['source'] and h['page'] == item['page'] for h in hits)
    correct += found
    print(f"{'PASS' if found else 'MISS'}  {item['question']}")
print(f'Recall@{args.k}: {correct}/{len(rows)} = {correct/len(rows):.3f}')
