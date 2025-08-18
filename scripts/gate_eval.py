#!/usr/bin/env python3
import argparse, json, os, re, sys

def fnum(x):
    if isinstance(x,str):
        m=re.search(r'([0-9.]+)',x); return float(m.group(1)) if m else None
    return float(x) if x is not None else None

def parse_strict(p):
    if not os.path.exists(p): return None
    d=json.load(open(p))
    v=(d.get('format_compliance_rate') or d.get('compliance_rate')
       or d.get('compliance') or d.get('format_compliance'))
    v=fnum(v)
    if v and v>1: v/=100.0
    return v

def extract_v2_or_best(d):
    # 1) よくあるトップレベル名を先に直取り（あれば即採用）
    for key in ['v2_complete_result','v2_result','V2-Complete','v2_complete','V2Complete']:
        if key in d and isinstance(d[key],dict):
            q=d[key].get('quality')
            if q is not None: return float(q)
    # 2) 配列 conditions/results から V2-Complete 最優先、その次に最大
    pools=d.get('conditions') or d.get('results') or []
    v2=None; best=None
    for c in pools if isinstance(pools,list) else []:
        if not isinstance(c,dict): continue
        name=(c.get('name') or c.get('condition') or '').lower()
        q=c.get('quality') if c.get('quality') is not None else c.get('qual')
        if q is None: continue
        q=float(q)
        if 'v2' in name and 'complete' in name: v2=q
        best = q if best is None else max(best,q)
    if v2 is not None: return v2
    if best is not None: return best
    # 3) 連想項目に quality があれば拾う
    for k,v in d.items():
        if isinstance(v,dict):
            q=v.get('quality') if v.get('quality') is not None else v.get('qual')
            if q is not None: return float(q)
    # 4) 最後の保険
    q=d.get('quality') if d.get('quality') is not None else d.get('qual')
    return float(q) if q is not None else None

def parse_ablation(p, log_path):
    fmt=qual=None
    if p and os.path.exists(p):
        try:
            d=json.load(open(p))
            fmt=(d.get('format_compliance_rate') or d.get('format_compliance')
                 or d.get('format_compliance_ratio') or d.get('compliance'))
            fmt=fnum(fmt)
            if fmt and fmt>1: fmt/=100.0
            qual=extract_v2_or_best(d)
        except Exception:
            pass
    # ログfallback: V2-Complete の「qual=...」か、表形式の最終列品質
    if (fmt is None or qual is None) and log_path and os.path.exists(log_path):
        txt=open(log_path,errors='ignore').read()
        if fmt is None:
            m=re.search(r'(?:Format\s+)?Compliance(?:\s+Rate)?:\s*([0-9.]+)',txt,re.I)
            if m:
                fmt=float(m.group(1))
                if fmt>1: fmt/=100.0
        if qual is None:
            m=(re.search(r'V2-Complete.*?qual[:=]\s*([0-9.]+)',txt,re.I|re.S)
               or re.search(r'V2-Complete\s+[0-9.]+\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)',txt,re.I)
               or re.search(r'\bQuality\s+([0-9.]+)',txt,re.I))
            if m: qual=float(m.group(1))
    return fmt,qual

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir', default=os.environ.get('RUN_DIR',''))
    ap.add_argument('--comp-thresh', type=float, default=float(os.environ.get('COMP_THRESH','0.95')))
    ap.add_argument('--qual-thresh', type=float, default=float(os.environ.get('QUAL_THRESH','0.60')))
    ap.add_argument('--debug', action='store_true')
    args=ap.parse_args()

    r=args.run_dir
    if not r or not os.path.isdir(r):
        print(f"ERROR: RUN_DIR is invalid: {r}", file=sys.stderr)
        sys.exit(2)

    strict_json=os.path.join(r,'strict_report.json')
    ab_jsons=[os.path.join(r,'ablation_report.json'),
              os.path.join(r,'ablation_study_report.json')]
    ab_json=next((p for p in ab_jsons if os.path.exists(p)), '')
    log_ab=os.path.join(r,'out.log')

    comp=parse_strict(strict_json)
    fmt_ab,qual=parse_ablation(ab_json,log_ab)
    if comp is None: comp=fmt_ab

    C=args.comp_thresh; Q=args.qual_thresh
    ok=(comp is not None and comp>=C) and (qual is not None and qual>=Q)

    def fmt(x): return 'N/A' if x is None else f'{x:.3f}'
    print(f"GATE | compliance={fmt(comp)} (>= {C}) | quality={fmt(qual)} (>= {Q}) | {'PASS' if ok else 'FAIL'}")
    if args.debug:
        print("DEBUG: checking files:")
        print(f"  - strict JSON : {strict_json} ({'exists' if os.path.exists(strict_json) else 'missing'})")
        print(f"  - ablation JSON: {ab_json or '(not found)'}")
        if ab_json: print(f"      exists? {'yes' if os.path.exists(ab_json) else 'no'}")
        print(f"  - log file    : {log_ab} ({'exists' if os.path.exists(log_ab) else 'missing'})")
    print(f"- strict   : {strict_json if os.path.exists(strict_json) else '(not found)'}")
    print(f"- ablation : {ab_json or '(not found)'}")
    print(f"- log      : {log_ab if os.path.exists(log_ab) else '(not found)'}")
    sys.exit(0 if ok else 2)

if __name__=='__main__':
    main()
