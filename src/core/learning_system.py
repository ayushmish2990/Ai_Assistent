"""
Learning System for AI Coding Assistant

This module provides reinforcement learning and continuous improvement capabilities:
- Experience collection and analysis
- Performance feedback integration
- Adaptive behavior optimization
- Knowledge base evolution
- Success pattern recognition
- Failure analysis and prevention

Author: AI Coding Assistant
Version: 1.0.0
"""

import os
import json
import pickle
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
from pathlib import Path
import hashlib
from abc import ABC, abstractmethod
import statistics
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types of actions the agent can perform."""
    CODE_GENERATION = "code_generation"
    CODE_REFACTORING = "code_refactoring"
    BUG_FIXING = "bug_fixing"
    TEST_GENERATION = "test_generation"
    DOCUMENTATION = "documentation"
    SECURITY_ANALYSIS = "security_analysis"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    DEPLOYMENT = "deployment"
    PROJECT_SETUP = "project_setup"


class OutcomeType(Enum):
    """Possible outcomes of actions."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    ERROR = "error"
    TIMEOUT = "timeout"


class LearningMode(Enum):
    """Learning modes for the system."""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    BALANCED = "balanced"


@dataclass
class Experience:
    """Represents a learning experience."""
    id: str
    action_type: ActionType
    context: Dict[str, Any]
    action_taken: Dict[str, Any]
    outcome: OutcomeType
    reward: float
    execution_time: float
    success_metrics: Dict[str, float]
    error_details: Optional[str] = None
    user_feedback: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Pattern:
    """Represents a learned pattern."""
    id: str
    pattern_type: str
    conditions: Dict[str, Any]
    actions: List[Dict[str, Any]]
    success_rate: float
    confidence: float
    usage_count: int
    last_used: datetime
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class LearningMetrics:
    """Learning system performance metrics."""
    total_experiences: int
    success_rate: float
    average_reward: float
    improvement_rate: float
    pattern_count: int
    exploration_rate: float
    learning_efficiency: float
    adaptation_speed: float
    created_at: datetime = field(default_factory=datetime.now)


class RewardCalculator:
    """Calculates rewards for different actions and outcomes."""
    
    def __init__(self):
        self.base_rewards = {
            OutcomeType.SUCCESS: 1.0,
            OutcomeType.PARTIAL_SUCCESS: 0.5,
            OutcomeType.FAILURE: -0.5,
            OutcomeType.ERROR: -1.0,
            OutcomeType.TIMEOUT: -0.8
        }
        
        self.action_multipliers = {
            ActionType.CODE_GENERATION: 1.2,
            ActionType.BUG_FIXING: 1.5,
            ActionType.SECURITY_ANALYSIS: 1.3,
            ActionType.PERFORMANCE_OPTIMIZATION: 1.4,
            ActionType.TEST_GENERATION: 1.1,
            ActionType.CODE_REFACTORING: 1.0,
            ActionType.DOCUMENTATION: 0.8,
            ActionType.DEPLOYMENT: 1.3,
            ActionType.PROJECT_SETUP: 1.0
        }
    
    def calculate_reward(
        self,
        action_type: ActionType,
        outcome: OutcomeType,
        execution_time: float,
        success_metrics: Dict[str, float],
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate reward for an action."""
        # Base reward from outcome
        base_reward = self.base_rewards[outcome]
        
        # Action type multiplier
        action_multiplier = self.action_multipliers.get(action_type, 1.0)
        
        # Time efficiency bonus/penalty
        time_factor = self._calculate_time_factor(execution_time, action_type)
        
        # Success metrics bonus
        metrics_bonus = self._calculate_metrics_bonus(success_metrics)
        
        # User feedback adjustment
        feedback_adjustment = self._calculate_feedback_adjustment(user_feedback)
        
        # Final reward calculation
        reward = (base_reward * action_multiplier * time_factor) + metrics_bonus + feedback_adjustment
        
        return max(-2.0, min(2.0, reward))  # Clamp between -2 and 2
    
    def _calculate_time_factor(self, execution_time: float, action_type: ActionType) -> float:
        """Calculate time efficiency factor."""
        # Expected times for different actions (in seconds)
        expected_times = {
            ActionType.CODE_GENERATION: 30.0,
            ActionType.BUG_FIXING: 60.0,
            ActionType.TEST_GENERATION: 20.0,
            ActionType.CODE_REFACTORING: 45.0,
            ActionType.SECURITY_ANALYSIS: 90.0,
            ActionType.PERFORMANCE_OPTIMIZATION: 120.0,
            ActionType.DOCUMENTATION: 15.0,
            ActionType.DEPLOYMENT: 180.0,
            ActionType.PROJECT_SETUP: 300.0
        }
        
        expected_time = expected_times.get(action_type, 60.0)
        
        if execution_time <= expected_time * 0.5:
            return 1.2  # Fast execution bonus
        elif execution_time <= expected_time:
            return 1.0  # Normal execution
        elif execution_time <= expected_time * 2:
            return 0.8  # Slow execution penalty
        else:
            return 0.6  # Very slow execution penalty
    
    def _calculate_metrics_bonus(self, success_metrics: Dict[str, float]) -> float:
        """Calculate bonus from success metrics."""
        if not success_metrics:
            return 0.0
        
        # Weight different metrics
        weights = {
            'code_quality': 0.3,
            'test_coverage': 0.2,
            'performance': 0.2,
            'security': 0.2,
            'maintainability': 0.1
        }
        
        bonus = 0.0
        for metric, value in success_metrics.items():
            weight = weights.get(metric, 0.1)
            # Normalize value to 0-1 range and apply weight
            normalized_value = max(0, min(1, value))
            bonus += weight * normalized_value
        
        return bonus * 0.5  # Scale bonus to reasonable range
    
    def _calculate_feedback_adjustment(self, user_feedback: Optional[Dict[str, Any]]) -> float:
        """Calculate adjustment based on user feedback."""
        if not user_feedback:
            return 0.0
        
        adjustment = 0.0
        
        # Rating-based adjustment
        if 'rating' in user_feedback:
            rating = user_feedback['rating']  # Assume 1-5 scale
            adjustment += (rating - 3) * 0.2  # Convert to -0.4 to +0.4 range
        
        # Specific feedback flags
        if user_feedback.get('helpful', False):
            adjustment += 0.3
        if user_feedback.get('accurate', False):
            adjustment += 0.2
        if user_feedback.get('efficient', False):
            adjustment += 0.1
        
        if user_feedback.get('unhelpful', False):
            adjustment -= 0.3
        if user_feedback.get('inaccurate', False):
            adjustment -= 0.4
        if user_feedback.get('inefficient', False):
            adjustment -= 0.2
        
        return adjustment


class PatternRecognizer:
    """Recognizes patterns in successful and failed actions."""
    
    def __init__(self):
        self.min_pattern_occurrences = 3
        self.min_confidence_threshold = 0.7
    
    def analyze_experiences(self, experiences: List[Experience]) -> List[Pattern]:
        """Analyze experiences to identify patterns."""
        patterns = []
        
        # Group experiences by action type and outcome
        grouped_experiences = self._group_experiences(experiences)
        
        for (action_type, outcome), exp_list in grouped_experiences.items():
            if len(exp_list) >= self.min_pattern_occurrences:
                pattern = self._extract_pattern(action_type, outcome, exp_list)
                if pattern and pattern.confidence >= self.min_confidence_threshold:
                    patterns.append(pattern)
        
        return patterns
    
    def _group_experiences(self, experiences: List[Experience]) -> Dict[Tuple[ActionType, OutcomeType], List[Experience]]:
        """Group experiences by action type and outcome."""
        groups = defaultdict(list)
        
        for exp in experiences:
            key = (exp.action_type, exp.outcome)
            groups[key].append(exp)
        
        return groups
    
    def _extract_pattern(
        self,
        action_type: ActionType,
        outcome: OutcomeType,
        experiences: List[Experience]
    ) -> Optional[Pattern]:
        """Extract a pattern from similar experiences."""
        if not experiences:
            return None
        
        # Find common context elements
        common_context = self._find_common_context(experiences)
        
        # Find common action elements
        common_actions = self._find_common_actions(experiences)
        
        # Calculate success rate and confidence
        success_rate = len([e for e in experiences if e.outcome == OutcomeType.SUCCESS]) / len(experiences)
        confidence = self._calculate_pattern_confidence(experiences, common_context, common_actions)
        
        if confidence < self.min_confidence_threshold:
            return None
        
        pattern_id = hashlib.md5(
            f"{action_type.value}_{outcome.value}_{str(common_context)}_{str(common_actions)}".encode()
        ).hexdigest()[:12]
        
        return Pattern(
            id=pattern_id,
            pattern_type=f"{action_type.value}_{outcome.value}",
            conditions=common_context,
            actions=common_actions,
            success_rate=success_rate,
            confidence=confidence,
            usage_count=len(experiences),
            last_used=max(exp.timestamp for exp in experiences)
        )
    
    def _find_common_context(self, experiences: List[Experience]) -> Dict[str, Any]:
        """Find common context elements across experiences."""
        if not experiences:
            return {}
        
        # Start with first experience context
        common = dict(experiences[0].context)
        
        # Find intersection with other experiences
        for exp in experiences[1:]:
            keys_to_remove = []
            for key, value in common.items():
                if key not in exp.context or exp.context[key] != value:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del common[key]
        
        return common
    
    def _find_common_actions(self, experiences: List[Experience]) -> List[Dict[str, Any]]:
        """Find common action elements across experiences."""
        if not experiences:
            return []
        
        # Extract action sequences
        action_sequences = [exp.action_taken for exp in experiences]
        
        # Find most common action patterns
        # This is a simplified version - could be enhanced with sequence mining
        common_actions = []
        
        # Find actions that appear in most experiences
        action_counts = defaultdict(int)
        for actions in action_sequences:
            if isinstance(actions, dict):
                for key, value in actions.items():
                    action_counts[(key, str(value))] += 1
        
        threshold = len(experiences) * 0.7  # Must appear in 70% of experiences
        for (key, value), count in action_counts.items():
            if count >= threshold:
                common_actions.append({key: value})
        
        return common_actions
    
    def _calculate_pattern_confidence(
        self,
        experiences: List[Experience],
        common_context: Dict[str, Any],
        common_actions: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in the pattern."""
        if not experiences:
            return 0.0
        
        # Base confidence from number of experiences
        base_confidence = min(1.0, len(experiences) / 10.0)
        
        # Context consistency bonus
        context_consistency = len(common_context) / max(1, len(experiences[0].context))
        
        # Action consistency bonus
        action_consistency = len(common_actions) / max(1, len(experiences[0].action_taken))
        
        # Outcome consistency
        outcomes = [exp.outcome for exp in experiences]
        most_common_outcome = max(set(outcomes), key=outcomes.count)
        outcome_consistency = outcomes.count(most_common_outcome) / len(outcomes)
        
        # Weighted average
        confidence = (
            base_confidence * 0.3 +
            context_consistency * 0.2 +
            action_consistency * 0.2 +
            outcome_consistency * 0.3
        )
        
        return confidence


class AdaptiveBehaviorEngine:
    """Manages adaptive behavior based on learning."""
    
    def __init__(self):
        self.exploration_rate = 0.2
        self.learning_rate = 0.1
        self.decay_rate = 0.95
        self.patterns: Dict[str, Pattern] = {}
        self.action_values: Dict[str, float] = defaultdict(float)
        self.action_counts: Dict[str, int] = defaultdict(int)
    
    def select_action(
        self,
        action_type: ActionType,
        context: Dict[str, Any],
        available_actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select the best action based on learned patterns."""
        # Check for matching patterns
        matching_patterns = self._find_matching_patterns(action_type, context)
        
        if matching_patterns and np.random.random() > self.exploration_rate:
            # Exploitation: use best known pattern
            best_pattern = max(matching_patterns, key=lambda p: p.success_rate * p.confidence)
            return self._select_action_from_pattern(best_pattern, available_actions)
        else:
            # Exploration: try new actions or random selection
            return self._explore_action(available_actions)
    
    def update_from_experience(self, experience: Experience):
        """Update behavior based on new experience."""
        action_key = self._get_action_key(experience.action_type, experience.action_taken)
        
        # Update action values using Q-learning style update
        old_value = self.action_values[action_key]
        self.action_values[action_key] = old_value + self.learning_rate * (experience.reward - old_value)
        
        # Update action counts
        self.action_counts[action_key] += 1
        
        # Decay exploration rate
        self.exploration_rate *= self.decay_rate
        self.exploration_rate = max(0.05, self.exploration_rate)  # Minimum exploration
    
    def add_pattern(self, pattern: Pattern):
        """Add a learned pattern to the behavior engine."""
        self.patterns[pattern.id] = pattern
    
    def _find_matching_patterns(
        self,
        action_type: ActionType,
        context: Dict[str, Any]
    ) -> List[Pattern]:
        """Find patterns that match the current context."""
        matching = []
        
        for pattern in self.patterns.values():
            if pattern.pattern_type.startswith(action_type.value):
                # Check if context matches pattern conditions
                match_score = self._calculate_context_match(context, pattern.conditions)
                if match_score > 0.7:  # Threshold for considering a match
                    matching.append(pattern)
        
        return matching
    
    def _calculate_context_match(
        self,
        current_context: Dict[str, Any],
        pattern_conditions: Dict[str, Any]
    ) -> float:
        """Calculate how well current context matches pattern conditions."""
        if not pattern_conditions:
            return 0.0
        
        matches = 0
        total = len(pattern_conditions)
        
        for key, value in pattern_conditions.items():
            if key in current_context and current_context[key] == value:
                matches += 1
        
        return matches / total
    
    def _select_action_from_pattern(
        self,
        pattern: Pattern,
        available_actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select action based on a learned pattern."""
        # Try to find an action that matches the pattern
        for action in available_actions:
            for pattern_action in pattern.actions:
                if self._actions_similar(action, pattern_action):
                    return action
        
        # If no exact match, return first available action
        return available_actions[0] if available_actions else {}
    
    def _explore_action(self, available_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select an action for exploration."""
        if not available_actions:
            return {}
        
        # Use epsilon-greedy with action values
        action_scores = []
        for action in available_actions:
            action_key = self._get_action_key_from_dict(action)
            score = self.action_values.get(action_key, 0.0)
            # Add exploration bonus for less tried actions
            exploration_bonus = 1.0 / max(1, self.action_counts.get(action_key, 0))
            action_scores.append(score + exploration_bonus)
        
        # Select action with highest score
        best_idx = np.argmax(action_scores)
        return available_actions[best_idx]
    
    def _actions_similar(self, action1: Dict[str, Any], action2: Dict[str, Any]) -> bool:
        """Check if two actions are similar."""
        # Simple similarity check - could be enhanced
        common_keys = set(action1.keys()) & set(action2.keys())
        if not common_keys:
            return False
        
        matches = sum(1 for key in common_keys if action1[key] == action2[key])
        return matches / len(common_keys) > 0.8
    
    def _get_action_key(self, action_type: ActionType, action: Dict[str, Any]) -> str:
        """Generate a key for an action."""
        return f"{action_type.value}_{hash(str(sorted(action.items())))}"
    
    def _get_action_key_from_dict(self, action: Dict[str, Any]) -> str:
        """Generate a key from action dictionary."""
        return f"action_{hash(str(sorted(action.items())))}"


class LearningDatabase:
    """Manages persistent storage of learning data."""
    
    def __init__(self, db_path: str = "learning_data.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the learning database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id TEXT PRIMARY KEY,
                    action_type TEXT NOT NULL,
                    context TEXT NOT NULL,
                    action_taken TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    reward REAL NOT NULL,
                    execution_time REAL NOT NULL,
                    success_metrics TEXT,
                    error_details TEXT,
                    user_feedback TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    conditions TEXT NOT NULL,
                    actions TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    confidence REAL NOT NULL,
                    usage_count INTEGER NOT NULL,
                    last_used TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_experiences INTEGER NOT NULL,
                    success_rate REAL NOT NULL,
                    average_reward REAL NOT NULL,
                    improvement_rate REAL NOT NULL,
                    pattern_count INTEGER NOT NULL,
                    exploration_rate REAL NOT NULL,
                    learning_efficiency REAL NOT NULL,
                    adaptation_speed REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
    
    def save_experience(self, experience: Experience):
        """Save an experience to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO experiences 
                (id, action_type, context, action_taken, outcome, reward, 
                 execution_time, success_metrics, error_details, user_feedback, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experience.id,
                experience.action_type.value,
                json.dumps(experience.context),
                json.dumps(experience.action_taken),
                experience.outcome.value,
                experience.reward,
                experience.execution_time,
                json.dumps(experience.success_metrics),
                experience.error_details,
                json.dumps(experience.user_feedback) if experience.user_feedback else None,
                experience.timestamp.isoformat()
            ))
    
    def load_experiences(
        self,
        limit: Optional[int] = None,
        action_type: Optional[ActionType] = None,
        since: Optional[datetime] = None
    ) -> List[Experience]:
        """Load experiences from the database."""
        query = "SELECT * FROM experiences"
        params = []
        conditions = []
        
        if action_type:
            conditions.append("action_type = ?")
            params.append(action_type.value)
        
        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        experiences = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            for row in cursor.fetchall():
                experiences.append(Experience(
                    id=row[0],
                    action_type=ActionType(row[1]),
                    context=json.loads(row[2]),
                    action_taken=json.loads(row[3]),
                    outcome=OutcomeType(row[4]),
                    reward=row[5],
                    execution_time=row[6],
                    success_metrics=json.loads(row[7]) if row[7] else {},
                    error_details=row[8],
                    user_feedback=json.loads(row[9]) if row[9] else None,
                    timestamp=datetime.fromisoformat(row[10])
                ))
        
        return experiences
    
    def save_pattern(self, pattern: Pattern):
        """Save a pattern to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO patterns 
                (id, pattern_type, conditions, actions, success_rate, 
                 confidence, usage_count, last_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.id,
                pattern.pattern_type,
                json.dumps(pattern.conditions),
                json.dumps(pattern.actions),
                pattern.success_rate,
                pattern.confidence,
                pattern.usage_count,
                pattern.last_used.isoformat(),
                pattern.created_at.isoformat()
            ))
    
    def load_patterns(self) -> List[Pattern]:
        """Load all patterns from the database."""
        patterns = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM patterns ORDER BY confidence DESC")
            for row in cursor.fetchall():
                patterns.append(Pattern(
                    id=row[0],
                    pattern_type=row[1],
                    conditions=json.loads(row[2]),
                    actions=json.loads(row[3]),
                    success_rate=row[4],
                    confidence=row[5],
                    usage_count=row[6],
                    last_used=datetime.fromisoformat(row[7]),
                    created_at=datetime.fromisoformat(row[8])
                ))
        
        return patterns
    
    def save_metrics(self, metrics: LearningMetrics):
        """Save learning metrics to the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO metrics 
                (total_experiences, success_rate, average_reward, improvement_rate,
                 pattern_count, exploration_rate, learning_efficiency, adaptation_speed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.total_experiences,
                metrics.success_rate,
                metrics.average_reward,
                metrics.improvement_rate,
                metrics.pattern_count,
                metrics.exploration_rate,
                metrics.learning_efficiency,
                metrics.adaptation_speed,
                metrics.created_at.isoformat()
            ))


class ContinuousLearningSystem:
    """Main learning system that coordinates all components."""
    
    def __init__(self, db_path: str = "learning_data.db"):
        self.database = LearningDatabase(db_path)
        self.reward_calculator = RewardCalculator()
        self.pattern_recognizer = PatternRecognizer()
        self.behavior_engine = AdaptiveBehaviorEngine()
        
        # Load existing patterns
        self._load_existing_patterns()
        
        # Learning parameters
        self.learning_window = timedelta(days=7)  # Analyze last 7 days
        self.pattern_update_interval = timedelta(hours=6)  # Update patterns every 6 hours
        self.last_pattern_update = datetime.now() - self.pattern_update_interval
    
    async def record_experience(
        self,
        action_type: ActionType,
        context: Dict[str, Any],
        action_taken: Dict[str, Any],
        outcome: OutcomeType,
        execution_time: float,
        success_metrics: Dict[str, float],
        error_details: Optional[str] = None,
        user_feedback: Optional[Dict[str, Any]] = None
    ) -> Experience:
        """Record a new learning experience."""
        # Calculate reward
        reward = self.reward_calculator.calculate_reward(
            action_type, outcome, execution_time, success_metrics, user_feedback
        )
        
        # Create experience
        experience = Experience(
            id=hashlib.md5(f"{datetime.now().isoformat()}_{action_type.value}".encode()).hexdigest()[:12],
            action_type=action_type,
            context=context,
            action_taken=action_taken,
            outcome=outcome,
            reward=reward,
            execution_time=execution_time,
            success_metrics=success_metrics,
            error_details=error_details,
            user_feedback=user_feedback
        )
        
        # Save to database
        self.database.save_experience(experience)
        
        # Update behavior engine
        self.behavior_engine.update_from_experience(experience)
        
        # Check if we should update patterns
        if datetime.now() - self.last_pattern_update > self.pattern_update_interval:
            await self._update_patterns()
        
        return experience
    
    async def get_recommended_action(
        self,
        action_type: ActionType,
        context: Dict[str, Any],
        available_actions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get recommended action based on learned patterns."""
        return self.behavior_engine.select_action(action_type, context, available_actions)
    
    async def analyze_performance(self, days: int = 7) -> LearningMetrics:
        """Analyze learning system performance."""
        since = datetime.now() - timedelta(days=days)
        experiences = self.database.load_experiences(since=since)
        
        if not experiences:
            return LearningMetrics(
                total_experiences=0,
                success_rate=0.0,
                average_reward=0.0,
                improvement_rate=0.0,
                pattern_count=0,
                exploration_rate=self.behavior_engine.exploration_rate,
                learning_efficiency=0.0,
                adaptation_speed=0.0
            )
        
        # Calculate metrics
        total_experiences = len(experiences)
        successful_experiences = [e for e in experiences if e.outcome == OutcomeType.SUCCESS]
        success_rate = len(successful_experiences) / total_experiences
        average_reward = statistics.mean([e.reward for e in experiences])
        
        # Calculate improvement rate (compare first and second half)
        mid_point = len(experiences) // 2
        if mid_point > 0:
            first_half_reward = statistics.mean([e.reward for e in experiences[mid_point:]])
            second_half_reward = statistics.mean([e.reward for e in experiences[:mid_point]])
            improvement_rate = (second_half_reward - first_half_reward) / abs(first_half_reward) if first_half_reward != 0 else 0
        else:
            improvement_rate = 0.0
        
        # Pattern count
        patterns = self.database.load_patterns()
        pattern_count = len(patterns)
        
        # Learning efficiency (reward per experience)
        learning_efficiency = average_reward / max(1, total_experiences)
        
        # Adaptation speed (how quickly rewards improve)
        adaptation_speed = self._calculate_adaptation_speed(experiences)
        
        metrics = LearningMetrics(
            total_experiences=total_experiences,
            success_rate=success_rate,
            average_reward=average_reward,
            improvement_rate=improvement_rate,
            pattern_count=pattern_count,
            exploration_rate=self.behavior_engine.exploration_rate,
            learning_efficiency=learning_efficiency,
            adaptation_speed=adaptation_speed
        )
        
        # Save metrics
        self.database.save_metrics(metrics)
        
        return metrics
    
    async def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the learning process."""
        experiences = self.database.load_experiences(limit=1000)
        patterns = self.database.load_patterns()
        
        insights = {
            "total_experiences": len(experiences),
            "total_patterns": len(patterns),
            "action_type_distribution": self._analyze_action_distribution(experiences),
            "outcome_distribution": self._analyze_outcome_distribution(experiences),
            "top_patterns": self._get_top_patterns(patterns),
            "learning_trends": self._analyze_learning_trends(experiences),
            "improvement_areas": self._identify_improvement_areas(experiences)
        }
        
        return insights
    
    def _load_existing_patterns(self):
        """Load existing patterns into the behavior engine."""
        patterns = self.database.load_patterns()
        for pattern in patterns:
            self.behavior_engine.add_pattern(pattern)
    
    async def _update_patterns(self):
        """Update patterns based on recent experiences."""
        since = datetime.now() - self.learning_window
        experiences = self.database.load_experiences(since=since)
        
        if len(experiences) >= self.pattern_recognizer.min_pattern_occurrences:
            new_patterns = self.pattern_recognizer.analyze_experiences(experiences)
            
            for pattern in new_patterns:
                self.database.save_pattern(pattern)
                self.behavior_engine.add_pattern(pattern)
        
        self.last_pattern_update = datetime.now()
    
    def _calculate_adaptation_speed(self, experiences: List[Experience]) -> float:
        """Calculate how quickly the system adapts to new situations."""
        if len(experiences) < 10:
            return 0.0
        
        # Calculate moving average of rewards
        window_size = min(10, len(experiences) // 4)
        rewards = [e.reward for e in reversed(experiences)]  # Most recent first
        
        moving_averages = []
        for i in range(len(rewards) - window_size + 1):
            window = rewards[i:i + window_size]
            moving_averages.append(statistics.mean(window))
        
        if len(moving_averages) < 2:
            return 0.0
        
        # Calculate rate of improvement in moving averages
        improvements = []
        for i in range(1, len(moving_averages)):
            if moving_averages[i-1] != 0:
                improvement = (moving_averages[i] - moving_averages[i-1]) / abs(moving_averages[i-1])
                improvements.append(improvement)
        
        return statistics.mean(improvements) if improvements else 0.0
    
    def _analyze_action_distribution(self, experiences: List[Experience]) -> Dict[str, int]:
        """Analyze distribution of action types."""
        distribution = defaultdict(int)
        for exp in experiences:
            distribution[exp.action_type.value] += 1
        return dict(distribution)
    
    def _analyze_outcome_distribution(self, experiences: List[Experience]) -> Dict[str, int]:
        """Analyze distribution of outcomes."""
        distribution = defaultdict(int)
        for exp in experiences:
            distribution[exp.outcome.value] += 1
        return dict(distribution)
    
    def _get_top_patterns(self, patterns: List[Pattern], limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing patterns."""
        sorted_patterns = sorted(patterns, key=lambda p: p.success_rate * p.confidence, reverse=True)
        
        top_patterns = []
        for pattern in sorted_patterns[:limit]:
            top_patterns.append({
                "id": pattern.id,
                "type": pattern.pattern_type,
                "success_rate": pattern.success_rate,
                "confidence": pattern.confidence,
                "usage_count": pattern.usage_count
            })
        
        return top_patterns
    
    def _analyze_learning_trends(self, experiences: List[Experience]) -> Dict[str, Any]:
        """Analyze learning trends over time."""
        if not experiences:
            return {}
        
        # Sort by timestamp
        sorted_experiences = sorted(experiences, key=lambda e: e.timestamp)
        
        # Calculate trends
        rewards = [e.reward for e in sorted_experiences]
        execution_times = [e.execution_time for e in sorted_experiences]
        
        return {
            "reward_trend": "improving" if len(rewards) > 1 and rewards[-1] > rewards[0] else "declining",
            "efficiency_trend": "improving" if len(execution_times) > 1 and execution_times[-1] < execution_times[0] else "declining",
            "recent_success_rate": len([e for e in sorted_experiences[-20:] if e.outcome == OutcomeType.SUCCESS]) / min(20, len(sorted_experiences))
        }
    
    def _identify_improvement_areas(self, experiences: List[Experience]) -> List[str]:
        """Identify areas that need improvement."""
        areas = []
        
        # Analyze by action type
        action_performance = defaultdict(list)
        for exp in experiences:
            action_performance[exp.action_type].append(exp.reward)
        
        for action_type, rewards in action_performance.items():
            avg_reward = statistics.mean(rewards)
            if avg_reward < 0.3:  # Threshold for poor performance
                areas.append(f"Improve {action_type.value} performance (avg reward: {avg_reward:.2f})")
        
        # Check exploration vs exploitation balance
        if self.behavior_engine.exploration_rate < 0.1:
            areas.append("Consider increasing exploration rate")
        elif self.behavior_engine.exploration_rate > 0.4:
            areas.append("Consider reducing exploration rate")
        
        return areas


# Factory functions
def create_learning_system(db_path: str = "learning_data.db") -> ContinuousLearningSystem:
    """Create a new learning system instance."""
    return ContinuousLearningSystem(db_path)


async def quick_performance_analysis(db_path: str = "learning_data.db", days: int = 7) -> LearningMetrics:
    """Perform quick performance analysis."""
    system = create_learning_system(db_path)
    return await system.analyze_performance(days)


__all__ = [
    # Enums
    "ActionType",
    "OutcomeType", 
    "LearningMode",
    
    # Data classes
    "Experience",
    "Pattern",
    "LearningMetrics",
    
    # Core classes
    "RewardCalculator",
    "PatternRecognizer",
    "AdaptiveBehaviorEngine",
    "LearningDatabase",
    "ContinuousLearningSystem",
    
    # Factory functions
    "create_learning_system",
    "quick_performance_analysis"
]