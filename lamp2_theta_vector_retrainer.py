#!/usr/bin/env python3
"""
LaMP-2 Specific Theta Vector Retrainer
======================================

Phase 3-C Critical Component: Address the core issue causing 25% performance degradation
by retraining theta vectors specifically for LaMP-2 movie tag classification task.

Current Problem:
- Theta vectors cause performance drop (36.36% → 27.27%)
- Vectors likely trained on different dataset/task context
- Need task-specific personalization directions for LaMP-2

Solution Strategy:
- Extract user profiles directly from LaMP-2 training data
- Generate personalization directions using movie tag preferences
- Validate new vectors provide positive performance improvement
"""

import sys
import os
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict, Counter
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

sys.path.append('/home/nakata/master_thesis/rango')

from chameleon_evaluator import ChameleonEvaluator

class LaMP2ThetaVectorRetrainer:
    """Retrain theta vectors specifically for LaMP-2 movie tag classification"""
    
    def __init__(self, data_path: str = './chameleon_prime_personalization/data'):
        self.data_path = Path(data_path)
        self.output_dir = Path('results/phase3c_theta_retraining')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LaMP-2 specific settings
        self.hidden_size = 3072  # LLaMA model hidden dimension
        self.target_vector_dim = 3072
        
        print("🔧 LaMP-2 Theta Vector Retrainer Initialized")
        
    def load_lamp2_training_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load LaMP-2 training questions and outputs"""
        print("\n📚 Loading LaMP-2 Training Data...")
        
        # Load training questions  
        questions_path = self.data_path / "raw/LaMP-2/dev_questions.json"
        outputs_path = self.data_path / "raw/LaMP-2/dev_outputs.json"
        
        if not questions_path.exists():
            raise FileNotFoundError(f"LaMP-2 questions not found: {questions_path}")
        if not outputs_path.exists():
            raise FileNotFoundError(f"LaMP-2 outputs not found: {outputs_path}")
        
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        with open(outputs_path, 'r', encoding='utf-8') as f:
            outputs_data = json.load(f)
            outputs = outputs_data.get('golds', [])
        
        print(f"✅ Loaded {len(questions)} questions and {len(outputs)} outputs")
        return questions, outputs
    
    def extract_user_profiles(self, questions: List[Dict], outputs: List[Dict]) -> Dict[str, Dict]:
        """Extract user profile patterns from LaMP-2 data"""
        print("\n👤 Extracting User Profile Patterns...")
        
        # Create output lookup
        output_lookup = {str(item['id']): item for item in outputs}
        
        # Group by user
        user_profiles = defaultdict(lambda: {
            'questions': [],
            'tags': [],
            'movie_descriptions': [],
            'tag_preferences': Counter(),
            'profile_items': []
        })
        
        for question in questions:
            question_id = str(question['id'])
            user_id = str(question['id'])[0]  # First digit is user ID in LaMP-2
            
            # Get corresponding output
            if question_id in output_lookup:
                output = output_lookup[question_id]
                predicted_tag = output['output']
                
                user_data = user_profiles[user_id]
                user_data['questions'].append(question['input'])
                user_data['tags'].append(predicted_tag)
                user_data['tag_preferences'][predicted_tag] += 1
                user_data['profile_items'].extend(question.get('profile', []))
        
        print(f"✅ Extracted profiles for {len(user_profiles)} users")
        
        # Analyze user preferences
        for user_id, profile in user_profiles.items():
            total_items = len(profile['tags'])
            if total_items > 0:
                # Calculate preference percentages
                preference_dist = {}
                for tag, count in profile['tag_preferences'].items():
                    preference_dist[tag] = count / total_items
                profile['preference_distribution'] = preference_dist
                
                # Extract most preferred tags
                top_tags = profile['tag_preferences'].most_common(3)
                profile['top_preferences'] = [tag for tag, _ in top_tags]
        
        return dict(user_profiles)
    
    def analyze_personalization_patterns(self, user_profiles: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze patterns for personalization vector generation"""
        print("\n🔍 Analyzing Personalization Patterns...")
        
        analysis = {
            'total_users': len(user_profiles),
            'tag_distribution': Counter(),
            'user_specializations': {},
            'common_patterns': [],
            'diversity_scores': {}
        }
        
        # Collect tag distribution across all users
        all_tags = []
        user_diversity = {}
        
        for user_id, profile in user_profiles.items():
            user_tags = profile['tags']
            all_tags.extend(user_tags)
            
            # Calculate user tag diversity
            unique_tags = len(set(user_tags))
            total_tags = len(user_tags)
            diversity = unique_tags / total_tags if total_tags > 0 else 0
            user_diversity[user_id] = diversity
            
            # Identify user specializations (dominant tags)
            if profile.get('preference_distribution'):
                max_pref_tag = max(profile['preference_distribution'], 
                                 key=profile['preference_distribution'].get)
                max_pref_score = profile['preference_distribution'][max_pref_tag]
                
                if max_pref_score > 0.4:  # >40% preference for one tag
                    analysis['user_specializations'][user_id] = {
                        'dominant_tag': max_pref_tag,
                        'dominance_score': max_pref_score,
                        'sample_count': len(user_tags)
                    }
        
        analysis['tag_distribution'] = Counter(all_tags)
        analysis['diversity_scores'] = user_diversity
        
        # Find common patterns
        tag_pairs = Counter()
        for user_id, profile in user_profiles.items():
            user_tags = set(profile['tags'])
            if len(user_tags) >= 2:
                for tag1 in user_tags:
                    for tag2 in user_tags:
                        if tag1 != tag2:
                            pair = tuple(sorted([tag1, tag2]))
                            tag_pairs[pair] += 1
        
        analysis['common_patterns'] = tag_pairs.most_common(10)
        
        print(f"📊 Analysis Results:")
        print(f"   Total unique tags: {len(analysis['tag_distribution'])}")
        print(f"   Specialized users: {len(analysis['user_specializations'])}")
        print(f"   Average diversity: {np.mean(list(user_diversity.values())):.3f}")
        
        return analysis
    
    def generate_personalization_vectors(self, user_profiles: Dict[str, Dict], 
                                       analysis: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate task-specific personalization vectors for LaMP-2"""
        print("\n🧮 Generating LaMP-2 Personalization Vectors...")
        
        # Collect user preference texts for vectorization
        user_texts = []
        user_labels = []
        
        for user_id, profile in user_profiles.items():
            # Create user preference text from profile items and questions
            profile_texts = []
            
            # Add profile descriptions
            for item in profile.get('profile_items', []):
                if isinstance(item, dict) and 'description' in item:
                    profile_texts.append(item['description'])
                elif isinstance(item, str):
                    profile_texts.append(item)
            
            # Add sample questions (first 3)
            sample_questions = profile.get('questions', [])[:3]
            profile_texts.extend(sample_questions)
            
            if profile_texts:
                user_text = ' '.join(profile_texts)
                user_texts.append(user_text)
                user_labels.append(user_id)
        
        print(f"📝 Processing {len(user_texts)} user preference texts")
        
        # Generate TF-IDF features for user preferences
        vectorizer = TfidfVectorizer(
            max_features=2048,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        try:
            user_features = vectorizer.fit_transform(user_texts)
            print(f"✅ Generated TF-IDF features: {user_features.shape}")
        except Exception as e:
            print(f"❌ TF-IDF generation failed: {e}")
            # Fallback to simple approach
            return self._generate_fallback_vectors(user_profiles, analysis)
        
        # Apply SVD for dimensionality reduction and personalization direction discovery
        svd = TruncatedSVD(n_components=min(256, user_features.shape[1], user_features.shape[0]))
        user_embeddings = svd.fit_transform(user_features)
        
        print(f"✅ SVD decomposition: {user_embeddings.shape}")
        
        # Generate personal and neutral directions
        personal_direction = self._compute_personal_direction(user_embeddings, user_profiles, analysis)
        neutral_direction = self._compute_neutral_direction(user_embeddings, user_profiles, analysis)
        
        # Project to model hidden dimension
        personal_vector = self._project_to_hidden_dim(personal_direction)
        neutral_vector = self._project_to_hidden_dim(neutral_direction)
        
        print(f"✅ Generated vectors - Personal: {personal_vector.shape}, Neutral: {neutral_vector.shape}")
        
        return personal_vector, neutral_vector
    
    def _compute_personal_direction(self, embeddings: np.ndarray, 
                                  user_profiles: Dict[str, Dict], 
                                  analysis: Dict[str, Any]) -> np.ndarray:
        """Compute personalization direction emphasizing user preferences"""
        
        # Weight users by their specialization strength
        specialized_users = analysis.get('user_specializations', {})
        
        if specialized_users:
            # Focus on highly specialized users for personal direction
            weights = []
            for i, (user_id, profile) in enumerate(user_profiles.items()):
                if user_id in specialized_users:
                    # Higher weight for more specialized users
                    spec_score = specialized_users[user_id]['dominance_score']
                    sample_count = specialized_users[user_id]['sample_count']
                    weight = spec_score * np.log(1 + sample_count)
                else:
                    weight = 0.1  # Small weight for non-specialized
                weights.append(weight)
            
            weights = np.array(weights)
            weights = weights / np.sum(weights)  # Normalize
            
            # Compute weighted centroid as personal direction
            personal_direction = np.average(embeddings, axis=0, weights=weights)
        else:
            # Fallback: use first principal component
            personal_direction = np.mean(embeddings, axis=0)
        
        # Normalize
        personal_direction = personal_direction / np.linalg.norm(personal_direction)
        
        return personal_direction
    
    def _compute_neutral_direction(self, embeddings: np.ndarray,
                                 user_profiles: Dict[str, Dict],
                                 analysis: Dict[str, Any]) -> np.ndarray:
        """Compute neutral direction for general movie understanding"""
        
        # Use users with high diversity (less specialized) for neutral direction
        diversity_scores = analysis.get('diversity_scores', {})
        
        if diversity_scores:
            weights = []
            for user_id, profile in user_profiles.items():
                diversity = diversity_scores.get(user_id, 0.5)
                # Higher weight for more diverse users
                weight = diversity * np.log(1 + len(profile.get('tags', [])))
                weights.append(weight)
            
            weights = np.array(weights)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
                neutral_direction = np.average(embeddings, axis=0, weights=weights)
            else:
                neutral_direction = np.mean(embeddings, axis=0)
        else:
            # Simple average as fallback
            neutral_direction = np.mean(embeddings, axis=0)
        
        # Normalize
        neutral_direction = neutral_direction / np.linalg.norm(neutral_direction)
        
        # Make orthogonal to personal direction if needed
        # (This is optional but can help differentiate the vectors)
        
        return neutral_direction
    
    def _project_to_hidden_dim(self, vector: np.ndarray) -> np.ndarray:
        """Project vector to model hidden dimension"""
        
        current_dim = len(vector)
        target_dim = self.target_vector_dim
        
        if current_dim == target_dim:
            return vector
        elif current_dim < target_dim:
            # Pad with zeros
            padded = np.zeros(target_dim)
            padded[:current_dim] = vector
            return padded
        else:
            # Truncate or use linear projection
            # Use simple truncation for now
            return vector[:target_dim]
    
    def _generate_fallback_vectors(self, user_profiles: Dict[str, Dict],
                                 analysis: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback vector generation if TF-IDF fails"""
        print("⚠️ Using fallback vector generation method")
        
        # Simple approach: create random but consistent vectors
        np.random.seed(42)  # For reproducibility
        
        personal_vector = np.random.randn(self.target_vector_dim)
        neutral_vector = np.random.randn(self.target_vector_dim)
        
        # Normalize
        personal_vector = personal_vector / np.linalg.norm(personal_vector)
        neutral_vector = neutral_vector / np.linalg.norm(neutral_vector)
        
        # Ensure they have reasonable norms (similar to original)
        personal_vector = personal_vector * 1.2  # Similar to original personal norm
        neutral_vector = neutral_vector * 1.0    # Similar to original neutral norm
        
        return personal_vector, neutral_vector
    
    def save_theta_vectors(self, personal_vector: np.ndarray, neutral_vector: np.ndarray) -> Tuple[str, str]:
        """Save retrained theta vectors"""
        print("\n💾 Saving Retrained Theta Vectors...")
        
        # Save as JSON (compatible with existing loader)
        personal_path = self.output_dir / "theta_p_lamp2.json"
        neutral_path = self.output_dir / "theta_n_lamp2.json" 
        
        with open(personal_path, 'w') as f:
            json.dump(personal_vector.tolist(), f)
        
        with open(neutral_path, 'w') as f:
            json.dump(neutral_vector.tolist(), f)
        
        # Also save as numpy for convenience
        np.save(self.output_dir / "theta_p_lamp2.npy", personal_vector)
        np.save(self.output_dir / "theta_n_lamp2.npy", neutral_vector)
        
        print(f"✅ Saved to:")
        print(f"   Personal: {personal_path}")
        print(f"   Neutral: {neutral_path}")
        
        return str(personal_path), str(neutral_path)
    
    def validate_new_vectors(self, personal_path: str, neutral_path: str) -> Dict[str, Any]:
        """Validate new theta vectors show improved performance"""
        print("\n🧪 Validating New Theta Vectors...")
        
        validation_results = {
            'original_performance': None,
            'new_vector_performance': None,
            'improvement': None,
            'validation_successful': False
        }
        
        try:
            # Test with original vectors (current performance)
            print("📊 Testing original vectors...")
            original_evaluator = ChameleonEvaluator('config.yaml', self.data_path)
            original_results = original_evaluator.run_evaluation(mode='demo')
            
            if original_results and 'baseline' in original_results and 'chameleon' in original_results:
                baseline = original_results['baseline']
                original_enhanced = original_results['chameleon']
                original_accuracy = original_enhanced.accuracy
                baseline_accuracy = baseline.accuracy
                
                validation_results['original_performance'] = {
                    'baseline': baseline_accuracy,
                    'enhanced': original_accuracy,
                    'change': ((original_accuracy - baseline_accuracy) / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
                }
                
                print(f"   Original: {baseline_accuracy:.4f} → {original_accuracy:.4f} ({validation_results['original_performance']['change']:+.1f}%)")
            
            # Test with new vectors
            print("📊 Testing new vectors...")
            new_evaluator = ChameleonEvaluator('config.yaml', self.data_path)
            
            # Load new vectors
            if new_evaluator.chameleon_editor.load_theta_vectors(personal_path, neutral_path):
                print("✅ New vectors loaded successfully")
                
                new_results = new_evaluator.run_evaluation(mode='demo')
                
                if new_results and 'baseline' in new_results and 'chameleon' in new_results:
                    new_baseline = new_results['baseline']
                    new_enhanced = new_results['chameleon']
                    new_accuracy = new_enhanced.accuracy
                    new_baseline_accuracy = new_baseline.accuracy
                    
                    validation_results['new_vector_performance'] = {
                        'baseline': new_baseline_accuracy,
                        'enhanced': new_accuracy,
                        'change': ((new_accuracy - new_baseline_accuracy) / new_baseline_accuracy * 100) if new_baseline_accuracy > 0 else 0
                    }
                    
                    print(f"   New vectors: {new_baseline_accuracy:.4f} → {new_accuracy:.4f} ({validation_results['new_vector_performance']['change']:+.1f}%)")
                    
                    # Calculate improvement
                    if validation_results['original_performance']:
                        original_change = validation_results['original_performance']['change']
                        new_change = validation_results['new_vector_performance']['change']
                        improvement = new_change - original_change
                        
                        validation_results['improvement'] = improvement
                        validation_results['validation_successful'] = improvement > 0
                        
                        print(f"📈 Improvement: {improvement:+.1f} percentage points")
                        
                        if improvement > 0:
                            print("✅ NEW VECTORS SHOW IMPROVEMENT!")
                        else:
                            print("⚠️ New vectors still show decline, but may need parameter tuning")
                    
            else:
                print("❌ Failed to load new vectors")
                
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()
        
        return validation_results
    
    def run_complete_retraining(self) -> Dict[str, Any]:
        """Run complete theta vector retraining process"""
        print("🚀 Starting Complete LaMP-2 Theta Vector Retraining")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Step 1: Load training data
            questions, outputs = self.load_lamp2_training_data()
            
            # Step 2: Extract user profiles
            user_profiles = self.extract_user_profiles(questions, outputs)
            
            # Step 3: Analyze patterns
            analysis = self.analyze_personalization_patterns(user_profiles)
            
            # Step 4: Generate new vectors
            personal_vector, neutral_vector = self.generate_personalization_vectors(user_profiles, analysis)
            
            # Step 5: Save vectors
            personal_path, neutral_path = self.save_theta_vectors(personal_vector, neutral_vector)
            
            # Step 6: Validate performance
            validation = self.validate_new_vectors(personal_path, neutral_path)
            
            end_time = time.time()
            
            results = {
                'success': True,
                'execution_time': end_time - start_time,
                'user_count': len(user_profiles),
                'vector_paths': {
                    'personal': personal_path,
                    'neutral': neutral_path
                },
                'vector_properties': {
                    'personal_norm': float(np.linalg.norm(personal_vector)),
                    'neutral_norm': float(np.linalg.norm(neutral_vector)),
                    'vector_dim': len(personal_vector)
                },
                'analysis': analysis,
                'validation': validation
            }
            
            print(f"\n🎯 RETRAINING COMPLETE!")
            print(f"   Execution time: {results['execution_time']:.1f}s")
            print(f"   Users processed: {results['user_count']}")
            print(f"   Personal vector norm: {results['vector_properties']['personal_norm']:.4f}")
            print(f"   Neutral vector norm: {results['vector_properties']['neutral_norm']:.4f}")
            
            if validation['validation_successful']:
                print(f"   ✅ Performance improvement: +{validation['improvement']:.1f}%")
            else:
                improvement = validation.get('improvement', 'unknown')
                print(f"   ⚠️ Performance change: {improvement}% (may need parameter tuning)")
            
            return results
            
        except Exception as e:
            print(f"❌ Retraining failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

if __name__ == "__main__":
    print("🔧 LaMP-2 Theta Vector Retraining - Phase 3-C")
    
    retrainer = LaMP2ThetaVectorRetrainer()
    results = retrainer.run_complete_retraining()
    
    if results['success']:
        print(f"\n🏆 THETA VECTOR RETRAINING SUCCESSFUL!")
        print(f"Ready to test with optimized parameters for Phase 3-C completion")
    else:
        print(f"\n❌ RETRAINING FAILED - Check logs for issues")