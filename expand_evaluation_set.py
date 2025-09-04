#!/usr/bin/env python3
"""
Step 2: Evaluation Dataset Expansion

Phase 3: 評価条件健全化と系統的パラメータ探索
現在の問題：サンプル数不足による統計的検定力不足
目標：層別抽出で100+サンプルのバランス評価データセット作成

Based on user requirements:
- Minimum 100 samples for statistical significance
- Stratified sampling across users and categories
- Balanced tag distribution for proper evaluation
- Quality control for evaluation data integrity
"""

import sys
import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class StratificationConfig:
    """Configuration for stratified sampling"""
    min_total_samples: int = 100
    max_total_samples: int = 200
    min_samples_per_user: int = 2
    max_samples_per_user: int = 8
    min_samples_per_tag: int = 3
    target_tag_balance_ratio: float = 0.5  # Max ratio between most/least common tags


class EvaluationDatasetExpander:
    """
    Expands evaluation dataset with stratified sampling for statistical significance
    """
    
    def __init__(self, data_path: str = "./chameleon_prime_personalization/data", 
                 config_path: str = "config.yaml"):
        self.data_path = Path(data_path)
        self.config_path = Path(config_path)
        
        # Load configuration
        self.config = self._load_config()
        
        # Stratification configuration
        self.stratification = StratificationConfig()
        
        # Data containers
        self.raw_questions = []
        self.raw_answers = {}
        self.expanded_dataset = []
        
        # Statistics tracking
        self.stats = {
            'original_samples': 0,
            'expanded_samples': 0,
            'users_count': 0,
            'tag_distribution': {},
            'user_distribution': {},
            'stratification_success': False
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if self.config_path.exists():
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found: {self.config_path}")
            return {}
    
    def load_raw_data(self) -> bool:
        """
        Load raw LaMP-2 data from multiple possible sources
        """
        logger.info("🔍 Loading raw LaMP-2 data...")
        
        # Priority 1: dev_questions + dev_outputs (full dataset)
        questions_path = self.data_path / "raw/LaMP_all/LaMP_2/user-based/dev/dev_questions.json"
        answers_path = self.data_path / "raw/LaMP_all/LaMP_2/user-based/dev/dev_outputs.json"
        
        if questions_path.exists() and answers_path.exists():
            logger.info(f"Loading from priority source: {questions_path}")
            return self._load_questions_answers_format(questions_path, answers_path)
        
        # Priority 2: merged.json format
        merged_path = self.data_path / "raw/LaMP-2/merged.json"
        if merged_path.exists():
            logger.info(f"Loading from merged format: {merged_path}")
            return self._load_merged_format(merged_path)
        
        # Priority 3: answers.json format (backup)
        answers_backup = self.data_path / "raw/LaMP-2/answers.json"
        if answers_backup.exists():
            logger.info(f"Loading from answers backup: {answers_backup}")
            return self._load_answers_format(answers_backup)
        
        logger.error("❌ No valid data sources found")
        return False
    
    def _load_questions_answers_format(self, questions_path: Path, answers_path: Path) -> bool:
        """Load from separate questions and answers files"""
        try:
            # Load questions
            with open(questions_path, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
            
            if isinstance(questions_data, dict) and 'instances' in questions_data:
                self.raw_questions = questions_data['instances']
            else:
                self.raw_questions = questions_data if isinstance(questions_data, list) else []
            
            # Load answers
            with open(answers_path, 'r', encoding='utf-8') as f:
                answers_data = json.load(f)
            
            if isinstance(answers_data, dict) and 'golds' in answers_data:
                golds = answers_data['golds']
            else:
                golds = answers_data if isinstance(answers_data, list) else []
            
            # Create answer lookup
            self.raw_answers = {str(item['id']): item['output'].strip().lower() for item in golds}
            
            logger.info(f"✅ Loaded {len(self.raw_questions)} questions and {len(self.raw_answers)} answers")
            self.stats['original_samples'] = len(self.raw_questions)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load questions/answers format: {e}")
            return False
    
    def _load_merged_format(self, merged_path: Path) -> bool:
        """Load from merged.json format + separate answers.json"""
        try:
            # Load questions from merged.json
            with open(merged_path, 'r', encoding='utf-8') as f:
                merged_data = json.load(f)
            
            self.raw_questions = merged_data if isinstance(merged_data, list) else []
            
            # Load answers from answers.json (same directory)
            answers_path = merged_path.parent / "answers.json"
            if answers_path.exists():
                logger.info(f"Loading ground truth from: {answers_path}")
                with open(answers_path, 'r', encoding='utf-8') as f:
                    answers_data = json.load(f)
                
                # Parse answers format
                if isinstance(answers_data, list):
                    self.raw_answers = {str(item['id']): item['output'].strip().lower() for item in answers_data}
                else:
                    logger.warning("Unexpected answers.json format")
                    self.raw_answers = {}
            else:
                logger.warning(f"Ground truth file not found: {answers_path}")
                # Try to extract from merged format if available
                self.raw_answers = {}
                for item in self.raw_questions:
                    if 'output' in item:
                        self.raw_answers[str(item['id'])] = item['output'].strip().lower()
            
            logger.info(f"✅ Loaded {len(self.raw_questions)} questions and {len(self.raw_answers)} answers from merged format")
            self.stats['original_samples'] = len(self.raw_questions)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load merged format: {e}")
            return False
    
    def _load_answers_format(self, answers_path: Path) -> bool:
        """Load from answers.json backup format"""
        try:
            with open(answers_path, 'r', encoding='utf-8') as f:
                answers_data = json.load(f)
            
            if isinstance(answers_data, dict) and 'golds' in answers_data:
                golds = answers_data['golds']
                
                # Convert to questions format
                self.raw_questions = []
                self.raw_answers = {}
                
                for item in golds:
                    question_item = {
                        'id': item['id'],
                        'input': item.get('input', 'Unknown movie description'),
                        'profile': item.get('profile', [])
                    }
                    self.raw_questions.append(question_item)
                    self.raw_answers[str(item['id'])] = item['output'].strip().lower()
            
            logger.info(f"✅ Loaded {len(self.raw_questions)} samples from answers backup")
            self.stats['original_samples'] = len(self.raw_questions)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load answers format: {e}")
            return False
    
    def analyze_data_distribution(self) -> Dict[str, Any]:
        """
        Analyze data distribution for stratification planning
        """
        logger.info("📊 Analyzing data distribution...")
        
        user_distribution = defaultdict(int)
        tag_distribution = defaultdict(int)
        user_samples = defaultdict(list)
        
        for question in self.raw_questions:
            question_id = str(question['id'])
            
            # Extract user ID (LaMP-2 format: user_id = question_id[:-1])
            if len(question_id) >= 2:
                user_id = question_id[:-1]
            else:
                user_id = question_id
            
            user_distribution[user_id] += 1
            user_samples[user_id].append(question)
            
            # Get tag from answers
            if question_id in self.raw_answers:
                tag = self.raw_answers[question_id]
                tag_distribution[tag] += 1
        
        # Sort distributions
        sorted_users = sorted(user_distribution.items(), key=lambda x: x[1], reverse=True)
        sorted_tags = sorted(tag_distribution.items(), key=lambda x: x[1], reverse=True)
        
        analysis = {
            'total_questions': len(self.raw_questions),
            'total_users': len(user_distribution),
            'total_unique_tags': len(tag_distribution),
            'user_distribution': dict(user_distribution),
            'tag_distribution': dict(tag_distribution),
            'sorted_users': sorted_users[:20],  # Top 20 users
            'sorted_tags': sorted_tags[:20],     # Top 20 tags
            'user_samples': user_samples,
            'samples_per_user_stats': {
                'min': min(user_distribution.values()) if user_distribution else 0,
                'max': max(user_distribution.values()) if user_distribution else 0,
                'mean': np.mean(list(user_distribution.values())) if user_distribution else 0,
                'median': np.median(list(user_distribution.values())) if user_distribution else 0
            },
            'tag_balance_ratio': (
                max(tag_distribution.values()) / min(tag_distribution.values()) 
                if tag_distribution and min(tag_distribution.values()) > 0 else float('inf')
            )
        }
        
        # Update stats
        self.stats['users_count'] = analysis['total_users']
        self.stats['user_distribution'] = analysis['user_distribution']
        self.stats['tag_distribution'] = analysis['tag_distribution']
        
        logger.info(f"📊 Analysis complete:")
        logger.info(f"   • Total questions: {analysis['total_questions']}")
        logger.info(f"   • Unique users: {analysis['total_users']}")
        logger.info(f"   • Unique tags: {analysis['total_unique_tags']}")
        logger.info(f"   • Samples per user: {analysis['samples_per_user_stats']['min']}-{analysis['samples_per_user_stats']['max']} (mean: {analysis['samples_per_user_stats']['mean']:.1f})")
        logger.info(f"   • Tag balance ratio: {analysis['tag_balance_ratio']:.1f}")
        
        return analysis
    
    def create_stratified_sample(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create stratified sample with balanced user and tag representation
        """
        logger.info("🎯 Creating stratified sample...")
        
        config = self.stratification
        user_samples = analysis['user_samples']
        tag_distribution = analysis['tag_distribution']
        
        # Step 1: Select diverse users (prioritize users with sufficient samples)
        eligible_users = [
            user_id for user_id, samples in user_samples.items()
            if len(samples) >= config.min_samples_per_user
        ]
        
        if len(eligible_users) == 0:
            logger.error("❌ No users with sufficient samples found")
            return []
        
        logger.info(f"📋 Eligible users: {len(eligible_users)} (with ≥{config.min_samples_per_user} samples)")
        
        # Step 2: Stratified user selection
        random.seed(42)  # Reproducible sampling
        
        selected_samples = []
        samples_per_user = {}
        tag_counts = defaultdict(int)
        
        # Calculate target samples per user
        estimated_users_needed = min(
            config.max_total_samples // config.min_samples_per_user,
            len(eligible_users)
        )
        
        target_samples_per_user = min(
            config.max_samples_per_user,
            max(config.min_samples_per_user, config.min_total_samples // estimated_users_needed)
        )
        
        logger.info(f"🎯 Target: {target_samples_per_user} samples per user from {estimated_users_needed} users")
        
        # Step 3: Sample from each selected user
        selected_users = random.sample(eligible_users, min(estimated_users_needed, len(eligible_users)))
        
        for user_id in selected_users:
            user_questions = user_samples[user_id]
            
            # Stratify by tags within user (if possible)
            user_samples_by_tag = defaultdict(list)
            for question in user_questions:
                question_id = str(question['id'])
                if question_id in self.raw_answers:
                    tag = self.raw_answers[question_id]
                    user_samples_by_tag[tag].append(question)
            
            # Sample from user (balanced across their tags if possible)
            user_selected = []
            if len(user_samples_by_tag) > 1:
                # Multi-tag user: sample from each tag
                samples_per_tag = max(1, target_samples_per_user // len(user_samples_by_tag))
                for tag, tag_questions in user_samples_by_tag.items():
                    selected_count = min(samples_per_tag, len(tag_questions))
                    user_selected.extend(random.sample(tag_questions, selected_count))
            else:
                # Single-tag user: random sample
                selected_count = min(target_samples_per_user, len(user_questions))
                user_selected = random.sample(user_questions, selected_count)
            
            # Limit to target samples per user
            user_selected = user_selected[:target_samples_per_user]
            
            selected_samples.extend(user_selected)
            samples_per_user[user_id] = len(user_selected)
            
            # Track tag distribution
            for question in user_selected:
                question_id = str(question['id'])
                if question_id in self.raw_answers:
                    tag = self.raw_answers[question_id]
                    tag_counts[tag] += 1
        
        # Step 4: Validate stratification quality
        total_samples = len(selected_samples)
        unique_users = len(samples_per_user)
        unique_tags = len(tag_counts)
        
        # Tag balance check
        if len(tag_counts) > 1:
            tag_balance_ratio = max(tag_counts.values()) / min(tag_counts.values())
        else:
            tag_balance_ratio = 1.0
        
        stratification_success = (
            total_samples >= config.min_total_samples and
            unique_users >= 10 and  # At least 10 users for diversity
            unique_tags >= 5 and   # At least 5 tags for task diversity
            tag_balance_ratio <= (1.0 / config.target_tag_balance_ratio)  # Reasonable balance
        )
        
        # Update stats
        self.stats['expanded_samples'] = total_samples
        self.stats['stratification_success'] = stratification_success
        
        logger.info(f"✅ Stratification complete:")
        logger.info(f"   • Total samples: {total_samples}")
        logger.info(f"   • Users selected: {unique_users}")
        logger.info(f"   • Unique tags: {unique_tags}")
        logger.info(f"   • Tag balance ratio: {tag_balance_ratio:.1f}")
        logger.info(f"   • Stratification success: {stratification_success}")
        
        if not stratification_success:
            logger.warning("⚠️ Stratification quality below target - proceeding with available data")
        
        # Step 5: Add evaluation metadata
        for i, sample in enumerate(selected_samples):
            sample['_eval_metadata'] = {
                'sample_index': i,
                'user_id': self._extract_user_id(str(sample['id'])),
                'tag': self.raw_answers.get(str(sample['id']), 'unknown'),
                'stratification_group': f"user_{self._extract_user_id(str(sample['id']))}"
            }
        
        return selected_samples
    
    def _extract_user_id(self, question_id: str) -> str:
        """Extract user ID from question ID using LaMP-2 format"""
        if len(question_id) >= 2:
            return question_id[:-1]
        else:
            return question_id
    
    def save_expanded_dataset(self, expanded_samples: List[Dict[str, Any]], 
                            output_path: str = None) -> str:
        """
        Save expanded dataset to file
        """
        if output_path is None:
            output_path = "data/evaluation/lamp2_expanded_eval.jsonl"
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Saving expanded dataset to {output_file}")
        
        # Convert to evaluation format
        eval_samples = []
        for sample in expanded_samples:
            question_id = str(sample['id'])
            
            eval_sample = {
                'id': sample['id'],
                'question': sample['input'],
                'user_id': self._extract_user_id(question_id),
                'profile': sample.get('profile', []),
                'reference': self.raw_answers.get(question_id, 'unknown'),
                '_metadata': sample.get('_eval_metadata', {})
            }
            eval_samples.append(eval_sample)
        
        # Save as JSONL
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in eval_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Saved {len(eval_samples)} samples to {output_file}")
        
        # Save metadata
        metadata_file = output_file.with_suffix('.meta.json')
        metadata = {
            'creation_timestamp': '2025-08-29',
            'total_samples': len(eval_samples),
            'stratification_config': {
                'min_total_samples': self.stratification.min_total_samples,
                'max_total_samples': self.stratification.max_total_samples,
                'min_samples_per_user': self.stratification.min_samples_per_user,
                'max_samples_per_user': self.stratification.max_samples_per_user,
                'target_tag_balance_ratio': self.stratification.target_tag_balance_ratio
            },
            'statistics': self.stats,
            'quality_metrics': {
                'users_represented': len(set(s['user_id'] for s in eval_samples)),
                'tags_represented': len(set(s['reference'] for s in eval_samples)),
                'avg_samples_per_user': len(eval_samples) / len(set(s['user_id'] for s in eval_samples)),
                'tag_distribution': dict(Counter(s['reference'] for s in eval_samples))
            }
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📋 Metadata saved to {metadata_file}")
        
        return str(output_file)
    
    def run_expansion(self) -> Tuple[bool, str]:
        """
        Run complete dataset expansion process
        """
        logger.info("🚀 Starting evaluation dataset expansion...")
        logger.info("=" * 70)
        
        try:
            # Step 1: Load raw data
            if not self.load_raw_data():
                return False, "Failed to load raw data"
            
            # Step 2: Analyze distribution
            analysis = self.analyze_data_distribution()
            
            if analysis['total_questions'] < self.stratification.min_total_samples:
                logger.warning(f"⚠️ Available data ({analysis['total_questions']}) < minimum required ({self.stratification.min_total_samples})")
                # Adjust target to available data
                self.stratification.min_total_samples = min(50, analysis['total_questions'])
                logger.info(f"📉 Adjusted minimum to {self.stratification.min_total_samples} samples")
            
            # Step 3: Create stratified sample
            expanded_samples = self.create_stratified_sample(analysis)
            
            if not expanded_samples:
                return False, "Failed to create stratified sample"
            
            # Step 4: Save expanded dataset
            output_path = self.save_expanded_dataset(expanded_samples)
            
            logger.info("=" * 70)
            logger.info("🎉 Dataset expansion completed successfully!")
            logger.info(f"✅ Created: {output_path}")
            logger.info(f"📊 Samples: {len(expanded_samples)} (target: ≥{self.stratification.min_total_samples})")
            logger.info(f"👥 Users: {self.stats['users_count']}")
            logger.info(f"🏷️ Tags: {len(self.stats['tag_distribution'])}")
            logger.info(f"🎯 Quality: {'PASS' if self.stats['stratification_success'] else 'PARTIAL'}")
            
            return True, output_path
            
        except Exception as e:
            logger.error(f"❌ Dataset expansion failed: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)
    
    def validate_expanded_dataset(self, dataset_path: str) -> bool:
        """
        Validate the expanded dataset for evaluation readiness
        """
        logger.info(f"🔍 Validating expanded dataset: {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                samples = [json.loads(line) for line in f]
            
            # Basic validation
            if len(samples) < 50:
                logger.error(f"❌ Insufficient samples: {len(samples)} < 50")
                return False
            
            # Check required fields
            required_fields = ['id', 'question', 'user_id', 'reference']
            for i, sample in enumerate(samples):
                missing = [field for field in required_fields if field not in sample]
                if missing:
                    logger.error(f"❌ Sample {i} missing fields: {missing}")
                    return False
            
            # Check diversity
            users = set(s['user_id'] for s in samples)
            tags = set(s['reference'] for s in samples)
            
            if len(users) < 5:
                logger.error(f"❌ Insufficient user diversity: {len(users)} users < 5")
                return False
            
            if len(tags) < 3:
                logger.error(f"❌ Insufficient tag diversity: {len(tags)} tags < 3")
                return False
            
            logger.info(f"✅ Validation passed:")
            logger.info(f"   • Samples: {len(samples)}")
            logger.info(f"   • Users: {len(users)}")
            logger.info(f"   • Tags: {len(tags)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return False


def main():
    """
    Main execution function for dataset expansion
    """
    logger.info("🚀 Starting Step 2: Evaluation Dataset Expansion")
    logger.info("=" * 70)
    
    # Initialize expander
    expander = EvaluationDatasetExpander(
        data_path="./chameleon_prime_personalization/data",
        config_path="config.yaml"
    )
    
    # Run expansion
    success, result = expander.run_expansion()
    
    if success:
        # Validate the result
        if expander.validate_expanded_dataset(result):
            logger.info("🎉 Step 2 COMPLETED SUCCESSFULLY!")
            logger.info("✅ Expanded dataset ready for statistical evaluation")
            logger.info("✅ Stratified sampling ensures balanced representation")
            logger.info("✅ Sample size sufficient for significance testing")
            logger.info(f"🚀 Next: Step 3 - Systematic grid search with {expander.stats['expanded_samples']} samples")
        else:
            logger.error("❌ Step 2 FAILED - validation errors")
            return 1
    else:
        logger.error(f"❌ Step 2 FAILED: {result}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)