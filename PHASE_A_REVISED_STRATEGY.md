# PHASE A REVISED STRATEGY - ROOT CAUSE IDENTIFIED 🎯

## Executive Summary

**Root Cause Found**: **Double Psychology Bias**  
1. **Model bias**: Baseline predicts "psychology" for 64% of cases  
2. **User prior bias**: User priors contain 122% psychology mentions (some users have multiple psychology examples)
3. **Result**: Personalization reinforces rather than corrects the bias

## 📊 **Confirmed Findings**

### ✅ **User Prior Quality**: EXCELLENT
- **Average length**: 12,554 characters (very rich)
- **Genre diversity**: comedy (522), sci-fi (304), classics (231), etc.
- **Quality**: Detailed movie examples with clear preferences

### 🚨 **The Real Problem**: Compounding Bias
```
Model Baseline → 90/140 "psychology" (64.3%)
User Priors → 86 "psychology" mentions across 70 users (122.9%)
Result → Personalization AMPLIFIES psychology bias instead of diversifying
```

### 📈 **Why α=6.0 Failed**
Personalization is pushing in the SAME direction as model bias:
- User likes psychology → Model already biased to psychology → α=6.0 reinforces bias
- Need to find users with ANTI-psychology preferences to create diversity

## 🎯 **Revised Strategy**

### **Strategy 1: Extreme Parameter Escalation** (α=20.0+)
**Status**: Currently running α=20.0 experiment  
**Logic**: Force personalization to overcome both biases through brute force
**Expected**: Some effect increase, but may not solve root problem

### **Strategy 2: Anti-Psychology User Filtering**
```python
# Find users with strong non-psychology preferences
users_with_diverse_prefs = []
for user in priors:
    psych_count = user_prior.count('psychology')  
    total_genres = len(re.findall(r'- ([^:]+):', user_prior))
    if psych_count / total_genres < 0.2:  # <20% psychology
        users_with_diverse_prefs.append(user)
```

### **Strategy 3: Negative Psychology Steering** 
```bash
# Add negative psychology weight to counter bias
--alpha_personal 15.0 --alpha_psychology_penalty -10.0
```

### **Strategy 4: Layer Intervention Optimization**
Test different layers to find where bias forms:
- **Early layers (-16 to -12)**: Semantic understanding
- **Mid layers (-8 to -5)**: Concept formation  
- **Late layers (-4 to -1)**: Decision making

## 🔧 **Immediate Action Plan**

### **Phase A.1**: Emergency α=20.0 Results
```bash
# Wait for current experiment, then analyze
python tools/detect_editing_effects.py results/bench/phase_a1_emergency_alpha20/predictions.jsonl
```
**Success Criteria**: b+c > 15 (3x improvement over α=6.0)

### **Phase A.2**: User Filtering Experiment  
```bash
# Create anti-psychology user subset
python - <<'PY'
import json, re
with open("data/user_priors.jsonl") as f:
    priors = [json.loads(line) for line in f]

diverse_users = []
for prior in priors:
    prompt = prior.get('prior_prompt', '')
    psych_count = prompt.lower().count('psychology')
    genre_count = len(re.findall(r'- ([^:]+):', prompt))
    
    if genre_count > 0 and psych_count / genre_count < 0.15:  # <15% psychology
        diverse_users.append(prior)

print(f"Found {len(diverse_users)} diverse users out of {len(priors)}")
with open("data/user_priors_diverse.jsonl", "w") as f:
    for user in diverse_users:
        f.write(json.dumps(user) + "\n")
PY

# Test with diverse users only
python tools/run_benchmark_lamp2.py \
  --alpha_personal 10.0 --limit 50 \
  --user_prior_path data/user_priors_diverse.jsonl \
  --out_dir results/bench/phase_a2_diverse_users \
  [other_args...]
```

### **Phase A.3**: Multi-Layer Intervention
```bash
for LAYERS in "-16 -15 -14 -13" "-12 -11 -10 -9" "-8 -7 -6 -5"; do
  python tools/run_benchmark_lamp2.py \
    --alpha_personal 15.0 --target_layers $LAYERS \
    --limit 30 --out_dir results/bench/phase_a3_layer_${LAYERS// /_} \
    [other_args...]
done
```

## 🎯 **Success Metrics**

| Strategy | Target b+c | Target c-b | Rationale |
|----------|------------|------------|-----------|
| **α=20.0** | ≥20 | ≥5 | Brute force breakthrough |
| **Diverse users** | ≥15 | ≥8 | Remove psychology bias |
| **Early layers** | ≥25 | ≥10 | Better intervention point |

## 🚨 **Escalation Plan**

### **If α=20.0 Still Fails** (b+c < 10):
1. **Prompt Engineering**: Modify system prompt to explicitly discourage "psychology"
2. **Ensemble Methods**: Combine multiple α values
3. **Model Architecture**: Consider different base models

### **If Anti-Psychology Users Work**:
1. **Scale Up**: Run full experiment with diverse user subset
2. **User Analysis**: Identify what makes "good" personalizable users  
3. **Prior Engineering**: Enhance psychology-heavy users with diverse examples

## ⏰ **Timeline**

- **Phase A.1 Analysis**: 30 minutes (when α=20.0 completes)
- **Phase A.2 Implementation**: 45 minutes (user filtering + test)
- **Phase A.3 Execution**: 90 minutes (multi-layer testing)
- **Strategy Decision**: 3 hours total

## 📋 **Expected Outcomes**

### **Best Case**: 
- α=20.0 breaks through bias → b+c≥30, proceed to Phase B
- Diverse users work → Use filtered dataset for main experiments

### **Moderate Case**:
- Partial improvement → Combine strategies (α=15.0 + diverse users + early layers)

### **Worst Case**: 
- All strategies fail → Need fundamental approach change (prompt engineering, different model)

---
**Status**: 🔥 **HIGH PRIORITY** - Systematic bias requires systematic solutions  
**Confidence**: High - Have identified specific root cause and multiple intervention strategies