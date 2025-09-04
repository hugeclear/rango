#!/usr/bin/env python3
"""
Diagnose gate and direction vector health from predictions.

Usage:
    python tools/diagnose_gate_health.py results/bench/strict_n140/predictions.jsonl --output results/diagnostics/gate_health.md

Exit codes:
    0: Analysis completed successfully
    2: File not found or invalid
"""

import json
import sys
import argparse
import numpy as np
from collections import Counter
from pathlib import Path

def analyze_gate_stats(predictions):
    """Analyze gate application statistics"""
    gate_applied = []
    gate_values = []
    
    for pred in predictions:
        chameleon_info = pred.get("chameleon_info", {})
        if "gate_applied" in chameleon_info:
            gate_applied.append(chameleon_info["gate_applied"])
        if "gate_value" in chameleon_info:
            gate_values.append(chameleon_info["gate_value"])
    
    return {
        "gate_applied_rate": np.mean(gate_applied) if gate_applied else 0.0,
        "gate_applied_count": sum(gate_applied) if gate_applied else 0,
        "total_gates": len(gate_applied) if gate_applied else 0,
        "gate_values": gate_values
    }

def analyze_direction_vectors(predictions):
    """Analyze direction vector statistics"""
    l2_personal = []
    l2_general = []
    cos_theta = []
    
    for pred in predictions:
        chameleon_info = pred.get("chameleon_info", {})
        dv_stats = chameleon_info.get("direction_vector_stats", {})
        
        if "l2_personal" in dv_stats:
            l2_personal.append(dv_stats["l2_personal"])
        if "l2_general" in dv_stats:
            l2_general.append(dv_stats["l2_general"])
        if "cos_theta" in dv_stats:
            cos_theta.append(dv_stats["cos_theta"])
    
    def percentiles(data):
        if not data:
            return {"p25": 0, "p50": 0, "p75": 0}
        return {
            "p25": np.percentile(data, 25),
            "p50": np.percentile(data, 50), 
            "p75": np.percentile(data, 75)
        }
    
    return {
        "l2_personal": percentiles(l2_personal),
        "l2_general": percentiles(l2_general),
        "cos_theta": percentiles(cos_theta),
        "sample_count": len(l2_personal)
    }

def analyze_hook_registration(predictions):
    """Analyze hook registration statistics"""
    hook_counts = []
    target_layers_seen = set()
    
    for pred in predictions:
        chameleon_info = pred.get("chameleon_info", {})
        if "hooks_registered" in chameleon_info:
            hook_counts.append(chameleon_info["hooks_registered"])
        if "target_layers" in chameleon_info:
            target_layers_seen.update(chameleon_info["target_layers"])
    
    return {
        "hooks_registered": Counter(hook_counts),
        "target_layers": sorted(list(target_layers_seen)),
        "unique_hook_counts": len(set(hook_counts)) if hook_counts else 0
    }

def main():
    parser = argparse.ArgumentParser(description="Diagnose gate and direction vector health")
    parser.add_argument("predictions_file", help="Path to predictions.jsonl")
    parser.add_argument("--output", "-o", help="Output markdown file (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed stats")
    
    args = parser.parse_args()
    
    try:
        predictions = []
        with open(args.predictions_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        predictions.append(data)
                    except json.JSONDecodeError as e:
                        print(f"❌ Error on line {line_num}: {e}")
                        sys.exit(2)
        
        if not predictions:
            print(f"❌ No predictions found in {args.predictions_file}")
            sys.exit(2)
        
        # Analyze components
        gate_stats = analyze_gate_stats(predictions)
        dv_stats = analyze_direction_vectors(predictions)
        hook_stats = analyze_hook_registration(predictions)
        
        # Generate report
        report = []
        report.append("# Gate & Direction Vector Health Report")
        report.append("")
        report.append(f"- Dataset: `{args.predictions_file}`")
        report.append(f"- Total predictions: {len(predictions)}")
        report.append("")
        
        # Gate statistics
        report.append("## Gate Application Statistics")
        report.append(f"- Gate applied rate: **{gate_stats['gate_applied_rate']:.1%}**")
        report.append(f"- Gates applied: **{gate_stats['gate_applied_count']}/{gate_stats['total_gates']}**")
        if gate_stats['gate_values']:
            gate_vals = gate_stats['gate_values']
            report.append(f"- Gate value range: [{min(gate_vals):.4f}, {max(gate_vals):.4f}]")
        report.append("")
        
        # Direction vector statistics  
        report.append("## Direction Vector Statistics")
        if dv_stats['sample_count'] > 0:
            report.append(f"- Samples with DV stats: **{dv_stats['sample_count']}**")
            report.append("- L2 Personal:")
            for k, v in dv_stats['l2_personal'].items():
                report.append(f"  - {k}: {v:.6f}")
            report.append("- L2 General:")
            for k, v in dv_stats['l2_general'].items():
                report.append(f"  - {k}: {v:.6f}")
            report.append("- Cosine θ:")
            for k, v in dv_stats['cos_theta'].items():
                report.append(f"  - {k}: {v:.6f}")
        else:
            report.append("- **⚠️ No direction vector statistics found**")
        report.append("")
        
        # Hook registration
        report.append("## Hook Registration")
        if hook_stats['target_layers']:
            report.append(f"- Target layers: **{len(hook_stats['target_layers'])}**")
            report.append(f"  - {hook_stats['target_layers']}")
        else:
            report.append("- **⚠️ No target layer information found**")
        
        if hook_stats['hooks_registered']:
            report.append(f"- Hook counts: {dict(hook_stats['hooks_registered'])}")
        else:
            report.append("- **⚠️ No hook registration data found**")
        report.append("")
        
        # Health assessment
        report.append("## Health Assessment")
        issues = []
        if gate_stats['gate_applied_rate'] == 0:
            issues.append("🔴 No gates applied - check threshold or direction vectors")
        elif gate_stats['gate_applied_rate'] < 0.1:
            issues.append("🟡 Very low gate application rate")
        
        if dv_stats['sample_count'] == 0:
            issues.append("🔴 No direction vector statistics - check DV computation")
        elif all(v['p50'] == 0 for v in [dv_stats['l2_personal'], dv_stats['l2_general']]):
            issues.append("🔴 All direction vectors are zero - check DV generation")
        
        if not hook_stats['target_layers']:
            issues.append("🔴 No target layers detected - check hook registration")
        
        if not issues:
            issues.append("✅ All systems appear healthy")
        
        for issue in issues:
            report.append(f"- {issue}")
        
        report_content = "\n".join(report)
        
        # Output
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(report_content)
            print(f"📊 Report written to: {args.output}")
        else:
            print(report_content)
        
        sys.exit(0)
        
    except FileNotFoundError:
        print(f"❌ File not found: {args.predictions_file}")
        sys.exit(2)
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()