#!/usr/bin/env python3
"""
Temporal Constraints for Causal Chameleon Editing
Implements light cone temporal causality constraints

Ensures that Chameleon edits respect temporal ordering and causal precedence
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class TemporalEvent:
    """Represents a temporally-ordered event in user history"""
    timestamp: float
    event_type: str
    user_id: str
    content: str
    embedding: Optional[np.ndarray] = None

@dataclass
class CausalLightCone:
    """Light cone constraints for temporal causality"""
    past_events: List[TemporalEvent]
    future_horizon: float
    causality_radius: float
    max_influence_delay: float

class TemporalConstraintManager:
    """
    Manages temporal constraints for causal Chameleon editing
    
    Key principles:
    1. Light Cone Constraint: Events can only influence future events within causality radius
    2. Temporal Ordering: Cause must precede effect
    3. Influence Decay: Causal influence decays with temporal distance
    """
    
    def __init__(self, causality_radius: float = 86400.0,  # 24 hours in seconds
                 max_influence_delay: float = 604800.0,    # 7 days in seconds
                 influence_decay_rate: float = 0.1):
        """
        Initialize temporal constraint manager
        
        Args:
            causality_radius: Maximum time difference for direct causal influence (seconds)
            max_influence_delay: Maximum delay for any causal influence (seconds)
            influence_decay_rate: Exponential decay rate for temporal influence
        """
        self.causality_radius = causality_radius
        self.max_influence_delay = max_influence_delay
        self.influence_decay_rate = influence_decay_rate
        
        logger.info(f"Temporal constraints initialized: radius={causality_radius/3600:.1f}h, "
                   f"max_delay={max_influence_delay/86400:.1f}days")
    
    def extract_temporal_events(self, user_profiles: List[Dict]) -> List[TemporalEvent]:
        """
        Extract temporally-ordered events from user profiles
        
        Args:
            user_profiles: User profiles from LaMP-2 data
            
        Returns:
            List of TemporalEvent objects sorted by timestamp
        """
        events = []
        
        for profile in user_profiles:
            user_id = profile.get('user_id', 'unknown')
            
            for i, item in enumerate(profile.get('profile', [])):
                # Create pseudo-temporal ordering based on sequence
                base_timestamp = item.get('timestamp', i * 3600)  # 1 hour intervals if no timestamp
                
                event = TemporalEvent(
                    timestamp=float(base_timestamp),
                    event_type='interaction',
                    user_id=user_id,
                    content=item.get('description', ''),
                )
                events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        logger.info(f"Extracted {len(events)} temporal events from {len(user_profiles)} users")
        return events
    
    def build_light_cone(self, target_timestamp: float, 
                        all_events: List[TemporalEvent],
                        user_id: str) -> CausalLightCone:
        """
        Build light cone of causally relevant events
        
        Args:
            target_timestamp: Timestamp of target event
            all_events: All available temporal events
            user_id: User ID to focus on
            
        Returns:
            CausalLightCone with relevant past events
        """
        # Find past events within causality radius
        past_events = []
        
        for event in all_events:
            time_diff = target_timestamp - event.timestamp
            
            # Only consider past events
            if time_diff <= 0:
                continue
            
            # Only consider events within max influence delay
            if time_diff > self.max_influence_delay:
                continue
            
            # Prioritize same user events
            if event.user_id == user_id or time_diff <= self.causality_radius:
                past_events.append(event)
        
        # Sort by recency (most recent first)
        past_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return CausalLightCone(
            past_events=past_events,
            future_horizon=target_timestamp + self.causality_radius,
            causality_radius=self.causality_radius,
            max_influence_delay=self.max_influence_delay
        )
    
    def compute_temporal_influence_weights(self, light_cone: CausalLightCone, 
                                         target_timestamp: float) -> np.ndarray:
        """
        Compute temporal influence weights for events in light cone
        
        Args:
            light_cone: Light cone with relevant past events
            target_timestamp: Timestamp of target event
            
        Returns:
            Array of influence weights (higher = more influential)
        """
        if not light_cone.past_events:
            return np.array([])
        
        weights = []
        
        for event in light_cone.past_events:
            time_diff = target_timestamp - event.timestamp
            
            # Exponential decay with distance
            weight = np.exp(-self.influence_decay_rate * time_diff / 3600)  # Decay per hour
            
            # Boost weight for recent events within causality radius
            if time_diff <= self.causality_radius:
                weight *= 2.0
            
            weights.append(weight)
        
        return np.array(weights)
    
    def create_temporal_mask(self, hidden_states: torch.Tensor, 
                           light_cone: CausalLightCone,
                           target_timestamp: float) -> torch.Tensor:
        """
        Create temporal mask for constrained editing
        
        Args:
            hidden_states: Hidden states to be edited [batch_size, seq_len, hidden_dim]
            light_cone: Light cone with causal constraints
            target_timestamp: Current timestamp
            
        Returns:
            Temporal mask [batch_size, seq_len, hidden_dim] with causality constraints
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        # Start with all-ones mask (no constraints)
        temporal_mask = torch.ones_like(hidden_states)
        
        # Apply temporal influence weights
        influence_weights = self.compute_temporal_influence_weights(light_cone, target_timestamp)
        
        if len(influence_weights) == 0:
            return temporal_mask
        
        # Map influence weights to sequence positions
        # Assume more recent tokens correspond to more recent events
        for i in range(min(seq_len, len(influence_weights))):
            weight = influence_weights[i]
            # Apply weight to later tokens (more recent)
            pos = seq_len - 1 - i
            temporal_mask[:, pos, :] *= weight
        
        return temporal_mask
    
    def validate_causal_ordering(self, events: List[TemporalEvent]) -> Dict[str, Any]:
        """
        Validate that events maintain proper causal ordering
        
        Args:
            events: List of temporal events to validate
            
        Returns:
            Dictionary with validation results
        """
        violations = []
        total_pairs = 0
        
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events[i+1:], i+1):
                total_pairs += 1
                
                # Check temporal ordering
                if event1.timestamp > event2.timestamp:
                    violations.append({
                        'type': 'temporal_inversion',
                        'event1_idx': i,
                        'event2_idx': j,
                        'time_diff': event1.timestamp - event2.timestamp
                    })
                
                # Check causality radius violations
                time_diff = abs(event2.timestamp - event1.timestamp)
                if time_diff > self.max_influence_delay:
                    # Events too far apart to have causal influence
                    continue
        
        validation_result = {
            'total_events': len(events),
            'total_pairs': total_pairs,
            'violations': violations,
            'violation_rate': len(violations) / max(total_pairs, 1),
            'is_valid': len(violations) == 0
        }
        
        if violations:
            logger.warning(f"Temporal validation found {len(violations)} violations "
                          f"out of {total_pairs} event pairs ({validation_result['violation_rate']:.3f} rate)")
        
        return validation_result
    
    def apply_temporal_constraints_to_editing(self, 
                                            hidden_states: torch.Tensor,
                                            user_history: List[Dict],
                                            current_timestamp: Optional[float] = None) -> torch.Tensor:
        """
        Apply temporal constraints to Chameleon editing
        
        Args:
            hidden_states: Hidden states to constrain
            user_history: User history for temporal context
            current_timestamp: Current timestamp (default: now)
            
        Returns:
            Temporally-constrained hidden states
        """
        if current_timestamp is None:
            current_timestamp = datetime.now().timestamp()
        
        # Extract temporal events
        events = self.extract_temporal_events(user_history)
        
        if not events:
            logger.warning("No temporal events found, returning unconstrained states")
            return hidden_states
        
        # Build light cone
        user_id = user_history[0].get('user_id', 'unknown') if user_history else 'unknown'
        light_cone = self.build_light_cone(current_timestamp, events, user_id)
        
        # Create temporal mask
        temporal_mask = self.create_temporal_mask(hidden_states, light_cone, current_timestamp)
        
        # Apply constraints
        constrained_states = hidden_states * temporal_mask
        
        logger.debug(f"Applied temporal constraints using {len(light_cone.past_events)} past events")
        
        return constrained_states
    
    def get_causal_explanation(self, light_cone: CausalLightCone, 
                             target_timestamp: float) -> str:
        """
        Generate human-readable explanation of causal constraints
        
        Args:
            light_cone: Light cone with causal information
            target_timestamp: Target timestamp
            
        Returns:
            Textual explanation of causal relationships
        """
        if not light_cone.past_events:
            return "No causal precedents found within temporal constraints."
        
        influence_weights = self.compute_temporal_influence_weights(light_cone, target_timestamp)
        
        explanations = []
        for i, (event, weight) in enumerate(zip(light_cone.past_events[:5], influence_weights[:5])):
            time_ago = target_timestamp - event.timestamp
            time_str = f"{time_ago/3600:.1f}h ago" if time_ago < 86400 else f"{time_ago/86400:.1f}d ago"
            
            explanations.append(
                f"• {event.event_type} from {event.user_id} ({time_str}): "
                f"influence={weight:.3f}"
            )
        
        explanation = f"""Temporal Causal Analysis:
Target Time: {datetime.fromtimestamp(target_timestamp).strftime('%Y-%m-%d %H:%M:%S')}
Causality Radius: {self.causality_radius/3600:.1f} hours
Influential Past Events ({len(light_cone.past_events)} total):

""" + "\n".join(explanations)
        
        if len(light_cone.past_events) > 5:
            explanation += f"\n... and {len(light_cone.past_events) - 5} more events"
        
        return explanation

def integrate_temporal_constraints_with_chameleon(chameleon_editor, user_profiles: List[Dict],
                                                causality_radius: float = 86400.0) -> 'TemporalConstraintManager':
    """
    Integration helper for existing Chameleon system
    
    Args:
        chameleon_editor: Existing ChameleonEditor instance
        user_profiles: User profiles from data loader
        causality_radius: Temporal causality radius in seconds
        
    Returns:
        TemporalConstraintManager ready for integration
    """
    temporal_manager = TemporalConstraintManager(
        causality_radius=causality_radius,
        max_influence_delay=causality_radius * 7,  # 7x causality radius
        influence_decay_rate=0.1
    )
    
    # Validate temporal structure of user profiles
    events = temporal_manager.extract_temporal_events(user_profiles)
    validation_result = temporal_manager.validate_causal_ordering(events)
    
    if not validation_result['is_valid']:
        logger.warning(f"Temporal validation issues detected: {validation_result['violation_rate']:.3f} violation rate")
    
    logger.info(f"Temporal constraint manager integrated with {len(events)} events")
    return temporal_manager