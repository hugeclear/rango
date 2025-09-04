#!/usr/bin/env python3
"""
Causal-Constrained Chameleon Evaluator
Extends the base Chameleon system with causal inference constraints

Adds causal discovery, temporal constraints, and do-calculus to existing Chameleon editing
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np

from chameleon_evaluator import ChameleonEvaluator, EvaluationResult
from causal_inference import CausalGraphBuilder, TemporalConstraintManager, DoCalculusEstimator
from causal_inference.causal_graph_builder import integrate_with_chameleon_data_loader
from causal_inference.do_calculus import integrate_do_calculus_with_chameleon_evaluation

logger = logging.getLogger(__name__)

class CausalConstrainedChameleon(ChameleonEvaluator):
    """
    Causal-Constrained Chameleon Evaluator
    
    Extends base Chameleon with:
    1. Causal graph discovery using PC algorithm
    2. Temporal light cone constraints
    3. Average Treatment Effect (ATE) estimation
    4. Causally-aware editing constraints
    """
    
    def __init__(self, config_path: str | None, data_path: str, 
                 decoding_mode: str = "greedy",
                 enable_causal_constraints: bool = True,
                 causality_radius: float = 86400.0,
                 causal_alpha: float = 0.05):
        """
        Initialize Causal-Constrained Chameleon
        
        Args:
            config_path: Path to configuration file
            data_path: Path to data directory
            decoding_mode: Generation mode ("greedy" or "sample")
            enable_causal_constraints: Enable causal constraint enforcement
            causality_radius: Temporal causality radius (seconds)
            causal_alpha: Significance level for causal discovery
        """
        # Initialize base Chameleon system
        super().__init__(config_path, data_path, decoding_mode)
        
        # Causal inference configuration
        self.enable_causal_constraints = enable_causal_constraints
        self.causality_radius = causality_radius
        self.causal_alpha = causal_alpha
        
        # Causal analysis results storage
        self.causal_results_history = []
        self.ate_results = []
        
        logger.info(f"CausalConstrainedChameleon initialized with causality_radius={causality_radius/3600:.1f}h")
    
    def run_causal_evaluation(self, mode: str = "full", 
                            alpha_override: float = None, 
                            beta_override: float = None,
                            layers_override: List[str] = None,
                            **kwargs) -> Dict[str, Any]:
        """
        Run evaluation with causal analysis integration
        
        Args:
            mode: Evaluation mode ("demo", "full", etc.)
            alpha_override: Override alpha parameter
            beta_override: Override beta parameter
            layers_override: Override target layers
            
        Returns:
            Extended evaluation results with causal analysis
        """
        logger.info(f"Starting causal-aware evaluation in {mode} mode")
        
        # Get user samples for causal analysis
        if mode == "demo":
            user_limit = 10
        elif mode == "full":
            user_limit = self.config['evaluation']['max_users']
        else:
            user_limit = 20
        
        test_samples = self.data_loader.get_user_samples(user_limit)
        
        # Extract user profiles for causal inference
        user_profiles = integrate_with_chameleon_data_loader(self.data_loader, user_limit)
        
        # Initialize causal inference if enabled
        causal_success = False
        if self.enable_causal_constraints:
            causal_success = self.chameleon_editor.initialize_causal_inference(
                user_profiles, 
                enable_causal_constraints=True,
                causality_radius=self.causality_radius
            )
            
            if causal_success:
                logger.info("Causal constraints enabled for evaluation")
            else:
                logger.warning("Causal constraints initialization failed - proceeding without constraints")
        
        # Run base evaluation
        start_time = time.time()
        results = super().run_evaluation(
            mode=mode,
            alpha_override=alpha_override,
            beta_override=beta_override,
            layers_override=layers_override,
            **kwargs
        )
        
        # Add causal analysis if successful initialization
        if causal_success and results:
            try:
                causal_analysis = self._perform_causal_analysis(results, user_profiles)
                results['causal_analysis'] = causal_analysis
                
                # Estimate ATE from evaluation history
                if len(self.causal_results_history) >= 10:
                    ate_result = self._estimate_treatment_effects()
                    if ate_result:
                        results['treatment_effects'] = {
                            'ate_estimate': ate_result.ate_estimate,
                            'confidence_interval': ate_result.confidence_interval,
                            'p_value': ate_result.p_value,
                            'effect_size': ate_result.effect_size,
                            'statistical_power': ate_result.statistical_power
                        }
                        
                        # Generate causal report
                        dose_response = self._analyze_dose_response()
                        causal_report = self._generate_causal_report(ate_result, dose_response)
                        results['causal_report'] = causal_report
                
            except Exception as e:
                logger.error(f"Causal analysis failed: {e}")
                results['causal_analysis_error'] = str(e)
        
        # Store results for longitudinal causal analysis
        self.causal_results_history.append({
            'timestamp': time.time(),
            'mode': mode,
            'alpha_personal': alpha_override or self.config['chameleon']['alpha_personal'],
            'alpha_neutral': beta_override or self.config['chameleon']['alpha_general'],
            'baseline_accuracy': results.get('baseline', {}).get('accuracy', 0),
            'chameleon_accuracy': results.get('chameleon', {}).get('accuracy', 0),
            'layers': layers_override or self.config['chameleon']['target_layers'],
            'causal_constraints_used': causal_success
        })
        
        evaluation_time = time.time() - start_time
        logger.info(f"Causal evaluation completed in {evaluation_time:.2f}s")
        
        return results
    
    def _perform_causal_analysis(self, evaluation_results: Dict[str, Any], 
                                user_profiles: List[Dict]) -> Dict[str, Any]:
        """
        Perform causal analysis on evaluation results
        
        Args:
            evaluation_results: Results from base evaluation
            user_profiles: User profiles for causal analysis
            
        Returns:
            Dictionary with causal analysis results
        """
        causal_analysis = {
            'timestamp': time.time(),
            'user_profiles_analyzed': len(user_profiles),
            'causality_radius_hours': self.causality_radius / 3600,
        }
        
        # Causal graph analysis
        if self.chameleon_editor.causal_graph_builder:
            try:
                # Analyze causal relationships in user preferences
                graph_result = self.chameleon_editor.causal_graph_builder.build_causal_graph(
                    user_profiles, user_id='aggregate'
                )
                
                if graph_result:
                    causal_analysis['causal_graph'] = {
                        'n_nodes': len(graph_result.node_names),
                        'n_edges': int(np.sum(graph_result.adjacency_matrix != 0)),
                        'discovery_time': graph_result.discovery_time,
                        'alpha_threshold': graph_result.alpha_threshold,
                        'edge_density': float(np.sum(graph_result.adjacency_matrix != 0)) / (len(graph_result.node_names) ** 2)
                    }
                    
                    # Get causal explanation
                    graph_description = self.chameleon_editor.causal_graph_builder.visualize_causal_graph(graph_result)
                    causal_analysis['graph_description'] = graph_description
                    
            except Exception as e:
                logger.error(f"Causal graph analysis failed: {e}")
                causal_analysis['causal_graph_error'] = str(e)
        
        # Temporal constraint analysis
        if self.chameleon_editor.temporal_constraint_manager:
            try:
                # Extract and validate temporal events
                temporal_events = self.chameleon_editor.temporal_constraint_manager.extract_temporal_events(user_profiles)
                validation_result = self.chameleon_editor.temporal_constraint_manager.validate_causal_ordering(temporal_events)
                
                causal_analysis['temporal_analysis'] = {
                    'total_events': validation_result['total_events'],
                    'temporal_violations': validation_result['violations'],
                    'violation_rate': validation_result['violation_rate'],
                    'temporal_validity': validation_result['is_valid']
                }
                
                # Generate temporal explanation if there are events
                if temporal_events:
                    current_time = time.time()
                    light_cone = self.chameleon_editor.temporal_constraint_manager.build_light_cone(
                        current_time, temporal_events, user_profiles[0].get('user_id', 'unknown')
                    )
                    
                    explanation = self.chameleon_editor.temporal_constraint_manager.get_causal_explanation(
                        light_cone, current_time
                    )
                    causal_analysis['temporal_explanation'] = explanation
                    
            except Exception as e:
                logger.error(f"Temporal analysis failed: {e}")
                causal_analysis['temporal_analysis_error'] = str(e)
        
        return causal_analysis
    
    def _estimate_treatment_effects(self) -> Optional[Any]:
        """
        Estimate Average Treatment Effect from evaluation history
        
        Returns:
            ATE result or None if insufficient data
        """
        try:
            # Use do-calculus estimator on evaluation history
            ate_result = integrate_do_calculus_with_chameleon_evaluation(self.causal_results_history)
            
            if ate_result:
                self.ate_results.append(ate_result)
                logger.info(f"ATE estimated: {ate_result.ate_estimate:.4f} (p={ate_result.p_value:.4f})")
            
            return ate_result
            
        except Exception as e:
            logger.error(f"ATE estimation failed: {e}")
            return None
    
    def _analyze_dose_response(self) -> Dict[str, Any]:
        """
        Analyze dose-response relationship from evaluation history
        
        Returns:
            Dose-response analysis results
        """
        try:
            estimator = DoCalculusEstimator()
            
            # Convert history to treatments and outcomes format
            treatments = estimator.extract_treatments_from_evaluation_history(self.causal_results_history)
            outcomes = estimator.extract_outcomes_from_results(self.causal_results_history)
            
            dose_response = estimator.estimate_dose_response_function(treatments, outcomes)
            
            return dose_response
            
        except Exception as e:
            logger.error(f"Dose-response analysis failed: {e}")
            return {'error': str(e)}
    
    def _generate_causal_report(self, ate_result: Any, dose_response: Dict[str, Any]) -> str:
        """
        Generate comprehensive causal analysis report
        
        Args:
            ate_result: ATE estimation result
            dose_response: Dose-response analysis result
            
        Returns:
            Formatted causal analysis report
        """
        try:
            estimator = DoCalculusEstimator()
            report = estimator.generate_causal_report(ate_result, dose_response)
            
            # Add system-specific context
            system_context = f"""
CAUSAL-CONSTRAINED CHAMELEON SYSTEM ANALYSIS
===========================================

System Configuration:
• Causality Radius: {self.causality_radius/3600:.1f} hours
• Causal Discovery Alpha: {self.causal_alpha}
• Temporal Constraints: {'Enabled' if self.enable_causal_constraints else 'Disabled'}
• Evaluation History: {len(self.causal_results_history)} experiments

Integration Status:
• Causal Graph Builder: {'Active' if self.chameleon_editor.causal_graph_builder else 'Inactive'}
• Temporal Constraint Manager: {'Active' if self.chameleon_editor.temporal_constraint_manager else 'Inactive'}
• Causal Mask: {'Applied' if self.chameleon_editor.use_causal_constraints else 'Not Applied'}

"""
            
            return system_context + report
            
        except Exception as e:
            logger.error(f"Causal report generation failed: {e}")
            return f"Causal report generation failed: {e}"
    
    def save_causal_results(self, output_dir: Optional[str] = None) -> str:
        """
        Save causal analysis results to files
        
        Args:
            output_dir: Optional output directory
            
        Returns:
            Path to saved results
        """
        if output_dir:
            save_dir = Path(output_dir)
        else:
            save_dir = self.output_dir / "causal_analysis"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Save evaluation history
        history_file = save_dir / f"causal_evaluation_history_{timestamp}.json"
        import json
        with open(history_file, 'w') as f:
            json.dump(self.causal_results_history, f, indent=2)
        
        # Save ATE results if available
        if self.ate_results:
            ate_file = save_dir / f"ate_results_{timestamp}.json"
            ate_data = []
            for ate_result in self.ate_results:
                ate_data.append({
                    'treatment_name': ate_result.treatment.treatment_name,
                    'ate_estimate': ate_result.ate_estimate,
                    'confidence_interval': ate_result.confidence_interval,
                    'p_value': ate_result.p_value,
                    'effect_size': ate_result.effect_size,
                    'statistical_power': ate_result.statistical_power,
                    'n_control': ate_result.n_control,
                    'n_treatment': ate_result.n_treatment
                })
            
            with open(ate_file, 'w') as f:
                json.dump(ate_data, f, indent=2)
        
        logger.info(f"Causal analysis results saved to {save_dir}")
        return str(save_dir)

# Command line interface for causal evaluation
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Causal-Constrained Chameleon Evaluation")
    parser.add_argument("--mode", choices=["demo", "full"], default="demo")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--data_path", type=str, default="./", help="Data directory path")
    parser.add_argument("--enable_causal", action="store_true", default=True, help="Enable causal constraints")
    parser.add_argument("--causality_radius", type=float, default=86400.0, help="Causality radius in seconds")
    parser.add_argument("--alpha", type=float, help="Alpha parameter override")
    parser.add_argument("--beta", type=float, help="Beta parameter override")
    parser.add_argument("--layers", type=str, help="Target layers (comma-separated)")
    parser.add_argument("--save_results", action="store_true", help="Save causal analysis results")
    
    args = parser.parse_args()
    
    # Parse layers
    target_layers = None
    if args.layers:
        target_layers = [layer.strip() for layer in args.layers.split(',')]
    
    # Initialize causal evaluator
    evaluator = CausalConstrainedChameleon(
        config_path=args.config,
        data_path=args.data_path,
        enable_causal_constraints=args.enable_causal,
        causality_radius=args.causality_radius
    )
    
    # Run causal evaluation
    results = evaluator.run_causal_evaluation(
        mode=args.mode,
        alpha_override=args.alpha,
        beta_override=args.beta,
        layers_override=target_layers
    )
    
    # Display results
    if 'causal_analysis' in results:
        print("\n" + "="*60)
        print("🔬 CAUSAL ANALYSIS RESULTS")
        print("="*60)
        
        causal_results = results['causal_analysis']
        print(f"User profiles analyzed: {causal_results.get('user_profiles_analyzed', 0)}")
        print(f"Causality radius: {causal_results.get('causality_radius_hours', 0):.1f} hours")
        
        if 'causal_graph' in causal_results:
            graph_info = causal_results['causal_graph']
            print(f"Causal graph: {graph_info['n_nodes']} nodes, {graph_info['n_edges']} edges")
            print(f"Edge density: {graph_info['edge_density']:.3f}")
        
        if 'temporal_analysis' in causal_results:
            temporal_info = causal_results['temporal_analysis']
            print(f"Temporal events: {temporal_info['total_events']}")
            print(f"Temporal violations: {len(temporal_info['temporal_violations'])}")
        
        if 'treatment_effects' in results:
            ate_info = results['treatment_effects']
            print(f"Average Treatment Effect: {ate_info['ate_estimate']:.4f}")
            print(f"95% CI: [{ate_info['confidence_interval'][0]:.4f}, {ate_info['confidence_interval'][1]:.4f}]")
            print(f"p-value: {ate_info['p_value']:.4f}")
    
    # Save results if requested
    if args.save_results:
        save_path = evaluator.save_causal_results()
        print(f"\nResults saved to: {save_path}")
    
    print("\n✅ Causal evaluation completed!")