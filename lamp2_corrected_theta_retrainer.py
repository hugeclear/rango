#!/usr/bin/env python3
"""
Corrected LaMP-2 Specific Theta Vector Retrainer
================================================

CRITICAL FIX: Correct user ID extraction logic for LaMP-2 data.
- User ID is the first 2-3 digits of question ID, not just the first digit
- Question 110 → User 11, Question 1234 → User 123, etc.
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
import re

sys.path.append('/home/nakata/master_thesis/rango')

from chameleon_evaluator import ChameleonEvaluator

class CorrectedLaMP2ThetaVectorRetrainer:
    """Retrain theta vectors with CORRECTED user ID extraction for LaMP-2"""
    
    def __init__(self, data_path: str = './chameleon_prime_personalization/data'):
        self.data_path = Path(data_path)
        self.output_dir = Path('results/phase3c_theta_corrected')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # LaMP-2 specific settings
        self.hidden_size = 3072  # LLaMA model hidden dimension
        self.target_vector_dim = 3072
        
        print("🔧 CORRECTED LaMP-2 Theta Vector Retrainer Initialized")
        
    def extract_user_id_from_question_id(self, question_id: str) -> str:
        """Extract user ID from question ID with CORRECTED logic
        
        LaMP-2 format analysis:
        - Question ID: 110, 111, 112 → User ID: 11 (first 2 digits)  
        - Question ID: 1234, 1235 → User ID: 123 (first 3 digits)
        - General rule: User ID is question_id[:-1] (remove last digit)
        """
        if len(question_id) >= 2:
            # Remove last digit to get user ID
            user_id = question_id[:-1]
            return user_id
        else:
            # Fallback for edge cases
            return question_id
    
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
        """Extract user profile patterns from LaMP-2 data with CORRECTED user ID logic"""
        print("\n👤 Extracting User Profile Patterns (CORRECTED)...")
        
        # Create output lookup
        output_lookup = {str(item['id']): item for item in outputs}
        
        # Group by user (CORRECTED LOGIC)
        user_profiles = defaultdict(lambda: {
            'questions': [],
            'tags': [],
            'movie_descriptions': [],
            'tag_preferences': Counter(),
            'profile_items': []
        })
        
        for question in questions:
            question_id = str(question['id'])
            user_id = self.extract_user_id_from_question_id(question_id)  # CORRECTED EXTRACTION
            
            # Get corresponding output
            if question_id in output_lookup:
                output = output_lookup[question_id]
                predicted_tag = output['output']
                
                user_data = user_profiles[user_id]
                user_data['questions'].append(question['input'])
                user_data['tags'].append(predicted_tag)
                user_data['tag_preferences'][predicted_tag] += 1
                user_data['profile_items'].extend(question.get('profile', []))
        
        print(f"✅ Extracted profiles for {len(user_profiles)} users (CORRECTED)")
        
        # Show user ID distribution for verification
        user_sizes = [(user_id, len(profile['questions'])) for user_id, profile in user_profiles.items()]
        user_sizes.sort(key=lambda x: int(x[0]) if x[0].isdigit() else 999)  # Sort by user ID
        
        print(f"📊 User Distribution (first 10):")
        for user_id, question_count in user_sizes[:10]:
            print(f"   User {user_id}: {question_count} questions")
        
        if len(user_sizes) > 10:
            print(f"   ... and {len(user_sizes) - 10} more users")
        
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
            
            # Calculate user tag diversity (Shannon entropy-like measure)
            unique_tags = len(set(user_tags))
            total_tags = len(user_tags)
            diversity = unique_tags / total_tags if total_tags > 0 else 0
            user_diversity[user_id] = diversity
            
            # Identify user specializations (dominant tags)
            if profile.get('preference_distribution'):
                max_pref_tag = max(profile['preference_distribution'], 
                                 key=profile['preference_distribution'].get)
                max_pref_score = profile['preference_distribution'][max_pref_tag]
                
                if max_pref_score > 0.3:  # >30% preference for one tag
                    analysis['user_specializations'][user_id] = {
                        'dominant_tag': max_pref_tag,
                        'dominance_score': max_pref_score,
                        'sample_count': len(user_tags)
                    }
        
        analysis['tag_distribution'] = Counter(all_tags)
        analysis['diversity_scores'] = user_diversity
        
        # Find common tag co-occurrence patterns
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
        print(f"   Most common tags: {[tag for tag, _ in analysis['tag_distribution'].most_common(5)]}")
        
        return analysis
    
    def generate_personalization_vectors(self, user_profiles: Dict[str, Dict], 
                                       analysis: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate task-specific personalization vectors for LaMP-2"""
        print("\n🧮 Generating LaMP-2 Personalization Vectors...")
        
        # Collect user preference texts for vectorization
        user_texts = []
        user_labels = []
        user_tag_preferences = []
        
        for user_id, profile in user_profiles.items():
            # Create comprehensive user preference text
            profile_texts = []
            
            # Add profile descriptions (user's viewing history)
            for item in profile.get('profile_items', []):
                if isinstance(item, dict) and 'description' in item:
                    profile_texts.append(item['description'])
                elif isinstance(item, str):
                    profile_texts.append(item)
            
            # Add sample questions/movie descriptions
            sample_questions = profile.get('questions', [])
            for question in sample_questions[:5]:  # First 5 questions
                # Extract movie description from question
                if 'description:' in question:
                    desc_start = question.find('description:') + len('description:')
                    description = question[desc_start:].strip()
                    if description:
                        profile_texts.append(description)
            
            # Create user text representation
            if profile_texts:
                user_text = ' '.join(profile_texts)
                user_texts.append(user_text)
                user_labels.append(user_id)
                
                # Store tag preferences for weighted vector generation
                tag_prefs = profile.get('preference_distribution', {})
                user_tag_preferences.append(tag_prefs)
        
        print(f"📝 Processing {len(user_texts)} user preference texts")
        
        # Generate TF-IDF features for user preferences
        vectorizer = TfidfVectorizer(
            max_features=1024,  # Reduced to avoid memory issues
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,  # More permissive max_df
            lowercase=True
        )
        
        try:
            user_features = vectorizer.fit_transform(user_texts)
            print(f"✅ Generated TF-IDF features: {user_features.shape}")
        except Exception as e:
            print(f"❌ TF-IDF generation failed: {e}")
            # Enhanced fallback with tag-based vectors
            return self._generate_tag_based_vectors(user_profiles, analysis)
        
        # Apply SVD for dimensionality reduction
        n_components = min(128, user_features.shape[1], user_features.shape[0])
        svd = TruncatedSVD(n_components=n_components)
        user_embeddings = svd.fit_transform(user_features)
        
        print(f"✅ SVD decomposition: {user_embeddings.shape}")
        
        # Generate personal and neutral directions using advanced methods
        personal_direction = self._compute_tag_weighted_personal_direction(
            user_embeddings, user_profiles, user_tag_preferences, analysis
        )
        neutral_direction = self._compute_diversity_weighted_neutral_direction(
            user_embeddings, user_profiles, analysis
        )
        
        # Project to model hidden dimension
        personal_vector = self._project_to_hidden_dim(personal_direction)
        neutral_vector = self._project_to_hidden_dim(neutral_direction)
        
        print(f"✅ Generated vectors - Personal: {personal_vector.shape}, Neutral: {neutral_vector.shape}")
        
        return personal_vector, neutral_vector
    
    def _compute_tag_weighted_personal_direction(self, embeddings: np.ndarray,
                                               user_profiles: Dict[str, Dict],
                                               user_tag_preferences: List[Dict],
                                               analysis: Dict[str, Any]) -> np.ndarray:
        """Compute personalization direction weighted by tag specialization"""
        
        # Focus on users with strong tag preferences for personal direction
        specialized_users = analysis.get('user_specializations', {})
        
        weights = []
        for i, (user_id, profile) in enumerate(user_profiles.items()):
            if user_id in specialized_users:
                # Weight by specialization strength and sample size
                spec_data = specialized_users[user_id]
                dominance = spec_data['dominance_score']
                sample_count = spec_data['sample_count']
                
                # Boost weight for users with strong, consistent preferences
                weight = dominance * np.log1p(sample_count)
            else:
                # Small weight for non-specialized users
                weight = 0.1 * np.log1p(len(profile.get('tags', [])))
                
            weights.append(weight)
        
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)  # Normalize
            personal_direction = np.average(embeddings, axis=0, weights=weights)
        else:
            personal_direction = np.mean(embeddings, axis=0)
        
        # Normalize
        personal_direction = personal_direction / (np.linalg.norm(personal_direction) + 1e-8)
        
        return personal_direction
    
    def _compute_diversity_weighted_neutral_direction(self, embeddings: np.ndarray,
                                                    user_profiles: Dict[str, Dict],
                                                    analysis: Dict[str, Any]) -> np.ndarray:
        """Compute neutral direction using diverse users"""
        
        diversity_scores = analysis.get('diversity_scores', {})
        
        weights = []
        for user_id, profile in user_profiles.items():
            diversity = diversity_scores.get(user_id, 0.5)
            sample_count = len(profile.get('tags', []))
            
            # Higher weight for more diverse users with sufficient samples
            weight = diversity * np.log1p(sample_count)
            weights.append(weight)
        
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            neutral_direction = np.average(embeddings, axis=0, weights=weights)
        else:
            neutral_direction = np.mean(embeddings, axis=0)
        
        # Normalize
        neutral_direction = neutral_direction / (np.linalg.norm(neutral_direction) + 1e-8)
        
        return neutral_direction
    
    def _generate_tag_based_vectors(self, user_profiles: Dict[str, Dict],
                                  analysis: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate vectors based on tag co-occurrence patterns (fallback method)"""
        print("⚠️ Using tag-based fallback vector generation")
        
        # Get all unique tags
        all_tags = list(analysis['tag_distribution'].keys())
        tag_to_idx = {tag: i for i, tag in enumerate(all_tags)}
        
        # Create user-tag matrices
        user_tag_matrix = []
        for user_id, profile in user_profiles.items():
            tag_counts = profile.get('tag_preferences', Counter())
            user_vector = np.zeros(len(all_tags))
            
            for tag, count in tag_counts.items():
                if tag in tag_to_idx:
                    user_vector[tag_to_idx[tag]] = count
                    
            # Normalize by user's total tag count
            total_count = sum(tag_counts.values())
            if total_count > 0:
                user_vector = user_vector / total_count
                
            user_tag_matrix.append(user_vector)
        
        user_tag_matrix = np.array(user_tag_matrix)
        
        if user_tag_matrix.shape[0] == 0:
            # Ultimate fallback
            return self._generate_random_vectors()
        
        # Use SVD to find principal directions in tag space
        if user_tag_matrix.shape[1] > 1:
            svd = TruncatedSVD(n_components=min(64, user_tag_matrix.shape[1]))
            tag_embeddings = svd.fit_transform(user_tag_matrix)
            
            # First component as personal direction (main variation)
            personal_direction = svd.components_[0] if len(svd.components_) > 0 else np.mean(user_tag_matrix, axis=0)
            # Second component as neutral direction
            neutral_direction = svd.components_[1] if len(svd.components_) > 1 else np.mean(user_tag_matrix, axis=0)
        else:
            personal_direction = np.mean(user_tag_matrix, axis=0)
            neutral_direction = np.mean(user_tag_matrix, axis=0)
        
        # Project to hidden dimension
        personal_vector = self._project_to_hidden_dim(personal_direction)
        neutral_vector = self._project_to_hidden_dim(neutral_direction)
        
        return personal_vector, neutral_vector
    
    def _generate_random_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Ultimate fallback: generate random but consistent vectors"""
        print("⚠️ Using random vector generation (ultimate fallback)")
        
        np.random.seed(42)  # For reproducibility
        
        personal_vector = np.random.randn(self.target_vector_dim)
        neutral_vector = np.random.randn(self.target_vector_dim)
        
        # Normalize and scale
        personal_vector = personal_vector / np.linalg.norm(personal_vector) * 1.2
        neutral_vector = neutral_vector / np.linalg.norm(neutral_vector) * 1.0
        
        return personal_vector, neutral_vector
    
    def _project_to_hidden_dim(self, vector: np.ndarray) -> np.ndarray:
        """Project vector to model hidden dimension"""
        
        current_dim = len(vector)
        target_dim = self.target_vector_dim
        
        if current_dim == target_dim:
            return vector
        elif current_dim < target_dim:
            # Pad with small random values instead of zeros
            padded = np.zeros(target_dim)
            padded[:current_dim] = vector
            # Add small random noise to remaining dimensions
            padded[current_dim:] = np.random.randn(target_dim - current_dim) * 0.01
            return padded
        else:
            # Use PCA-style projection instead of truncation
            # For simplicity, just truncate for now
            return vector[:target_dim]
    
    def save_theta_vectors(self, personal_vector: np.ndarray, neutral_vector: np.ndarray) -> Tuple[str, str]:
        """Save retrained theta vectors"""
        print("\n💾 Saving Corrected Retrained Theta Vectors...")
        
        # Save as JSON (compatible with existing loader)
        personal_path = self.output_dir / "theta_p_lamp2_corrected.json"
        neutral_path = self.output_dir / "theta_n_lamp2_corrected.json"
        
        with open(personal_path, 'w') as f:
            json.dump(personal_vector.tolist(), f)
        
        with open(neutral_path, 'w') as f:
            json.dump(neutral_vector.tolist(), f)
        
        # Also save as numpy for convenience
        np.save(self.output_dir / "theta_p_lamp2_corrected.npy", personal_vector)
        np.save(self.output_dir / "theta_n_lamp2_corrected.npy", neutral_vector)
        
        print(f"✅ Saved to:")
        print(f"   Personal: {personal_path}")
        print(f"   Neutral: {neutral_path}")
        
        return str(personal_path), str(neutral_path)
    
    def validate_new_vectors(self, personal_path: str, neutral_path: str) -> Dict[str, Any]:
        """Validate new theta vectors with the effective parameters"""
        print("\n🧪 Validating New Theta Vectors (with effective parameters)...")
        
        validation_results = {
            'original_performance': None,
            'new_vector_performance': None,
            'improvement': None,
            'validation_successful': False
        }
        
        try:
            # Test with original vectors at effective parameters (α=2.0, β=-0.5)
            print("📊 Testing original vectors with effective parameters...")
            original_evaluator = ChameleonEvaluator('config.yaml', self.data_path)
            
            # Set effective parameters from Phase 3-A analysis
            original_evaluator.config['chameleon']['alpha_personal'] = 2.0
            original_evaluator.config['chameleon']['alpha_general'] = -0.5
            
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
                
                print(f"   Original (α=2.0, β=-0.5): {baseline_accuracy:.4f} → {original_accuracy:.4f} ({validation_results['original_performance']['change']:+.1f}%)")
            
            # Test with new vectors at same effective parameters
            print("📊 Testing new vectors with effective parameters...")
            new_evaluator = ChameleonEvaluator('config.yaml', self.data_path)
            
            # Set same effective parameters
            new_evaluator.config['chameleon']['alpha_personal'] = 2.0
            new_evaluator.config['chameleon']['alpha_general'] = -0.5
            
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
                    
                    print(f"   New vectors (α=2.0, β=-0.5): {new_baseline_accuracy:.4f} → {new_accuracy:.4f} ({validation_results['new_vector_performance']['change']:+.1f}%)")
                    
                    # Calculate improvement
                    if validation_results['original_performance']:
                        original_change = validation_results['original_performance']['change']
                        new_change = validation_results['new_vector_performance']['change']
                        improvement = new_change - original_change
                        
                        validation_results['improvement'] = improvement
                        validation_results['validation_successful'] = improvement > 5  # Require >5% improvement
                        
                        print(f"📈 Vector improvement: {improvement:+.1f} percentage points")
                        
                        if improvement > 5:
                            print("🎉 NEW VECTORS SHOW SIGNIFICANT IMPROVEMENT!")
                        elif improvement > 0:
                            print("✅ NEW VECTORS SHOW IMPROVEMENT!")
                        else:
                            print("⚠️ New vectors need further optimization")
                    
            else:
                print("❌ Failed to load new vectors")
                
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()
        
        return validation_results
    
    def run_complete_retraining(self) -> Dict[str, Any]:
        """Run complete theta vector retraining process with CORRECTED user extraction"""
        print("🚀 Starting CORRECTED LaMP-2 Theta Vector Retraining")
        print("=" * 70)
        
        start_time = time.time()
        
        try:
            # Step 1: Load training data
            questions, outputs = self.load_lamp2_training_data()
            
            # Step 2: Extract user profiles (CORRECTED)
            user_profiles = self.extract_user_profiles(questions, outputs)
            
            # Step 3: Analyze patterns
            analysis = self.analyze_personalization_patterns(user_profiles)
            
            # Step 4: Generate new vectors
            personal_vector, neutral_vector = self.generate_personalization_vectors(user_profiles, analysis)
            
            # Step 5: Save vectors
            personal_path, neutral_path = self.save_theta_vectors(personal_vector, neutral_vector)
            
            # Step 6: Validate performance with effective parameters
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
            
            print(f"\n🎯 CORRECTED RETRAINING COMPLETE!")
            print(f"   Execution time: {results['execution_time']:.1f}s")
            print(f"   Users processed: {results['user_count']}")
            print(f"   Personal vector norm: {results['vector_properties']['personal_norm']:.4f}")
            print(f"   Neutral vector norm: {results['vector_properties']['neutral_norm']:.4f}")
            
            if validation['validation_successful']:
                print(f"   🎉 SIGNIFICANT performance improvement: +{validation['improvement']:.1f}%")
            elif validation.get('improvement', 0) > 0:
                print(f"   ✅ Performance improvement: +{validation['improvement']:.1f}%")
            else:
                improvement = validation.get('improvement', 'unknown')
                print(f"   ⚠️ Performance change: {improvement}% (may need further optimization)")
            
            return results
            
        except Exception as e:
            print(f"❌ Corrected retraining failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': time.time() - start_time
            }

if __name__ == "__main__":
    print("🔧 CORRECTED LaMP-2 Theta Vector Retraining - Phase 3-C")
    
    retrainer = CorrectedLaMP2ThetaVectorRetrainer()
    results = retrainer.run_complete_retraining()
    
    if results['success']:
        user_count = results['user_count']
        if user_count > 10:  # Should have many users now
            print(f"\n🎉 CORRECTED RETRAINING SUCCESSFUL!")
            print(f"   Processed {user_count} users (vs 1 user in original)")
            print(f"   Ready for Phase 3-C parameter optimization")
        else:
            print(f"\n⚠️ Still only {user_count} users - may need further investigation")
    else:
        print(f"\n❌ CORRECTED RETRAINING FAILED - Check logs for issues")