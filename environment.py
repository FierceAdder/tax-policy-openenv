from pydantic import BaseModel, Field
from typing import Tuple, Dict, Any

# 1. Pydantic Models for Typed Interfaces
class Observation(BaseModel):
    gdp: float = Field(..., description="Current Economic Output (GDP)")
    tax_rate: float = Field(..., description="Current Tax Rate in %")
    unemployment: float = Field(..., description="Unemployment rate in %")
    inequality: float = Field(..., description="Inequality index (0 to 1)")
    step_count: int = Field(..., description="Current year/step in the episode")

class Action(BaseModel):
    tax_change: float = Field(..., description="Change to the tax rate (e.g., -2.0 or +1.5)")

class Reward(BaseModel):
    value: float = Field(..., description="Reward value for the step")

# 2. OpenEnv Interface Implementation
class TaxPolicyEnv:
    def __init__(self, task: str = "easy"):
        self.task = task
        self.max_steps = 5  # <--- CHANGED FROM 10 TO 5
        self.reset()

    def reset(self) -> Observation:
        """Returns the initial observation."""
        self.current_step = 0
        self.gdp = 100.0
        self.tax_rate = 20.0
        self.unemployment = 5.0
        self.inequality = 0.50
        return self.state()

    def state(self) -> Observation:
        """Returns the current state."""
        return Observation(
            gdp=round(self.gdp, 2),
            tax_rate=round(self.tax_rate, 2),
            unemployment=round(self.unemployment, 2),
            inequality=round(self.inequality, 2),
            step_count=self.current_step
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Executes one step and returns (observation, reward, done, info)."""
        # Apply bounds to action to penalize infinite loops or destructive actions
        tax_change = max(min(action.tax_change, 5.0), -5.0) 
        
        # Update simulation variables
        self.tax_rate = max(min(self.tax_rate + tax_change, 80.0), 0.0)
        
        # Economic Logic: 
        # High tax slows GDP, increases unemployment, but lowers inequality
        gdp_growth = 5.0 - (self.tax_rate * 0.1)
        self.gdp += gdp_growth
        
        self.unemployment = max(2.0, 5.0 + (self.tax_rate - 20.0) * 0.15)
        self.inequality = max(0.1, min(0.9, 0.5 - (self.tax_rate - 20.0) * 0.01))
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        # Meaningful Reward Function mapping to tasks
        step_reward = self._calculate_reward(gdp_growth)
        
        info = {"grader_score": self._grade_episode()} if done else {}
        
        return self.state(), Reward(value=step_reward), done, info

    def _calculate_reward(self, gdp_growth: float) -> float:
        """Rewards incremental progress toward the objective."""
        if self.task == "easy":
            return round(gdp_growth * 0.2, 2)
        elif self.task == "medium":
            penalty = max(0, self.unemployment - 5.0) * 0.5
            return round((gdp_growth * 0.2) - penalty, 2)
        else: # hard
            unemp_penalty = max(0, self.unemployment - 5.0) * 0.5
            ineq_penalty = max(0, self.inequality - 0.5) * 5.0
            return round((gdp_growth * 0.2) - unemp_penalty - ineq_penalty, 2)

    def _grade_episode(self) -> float:
        """Programmatic grader assigning a deterministic score between 0.0 and 1.0."""
        # Base score on ending GDP (Baseline 100 -> Target 130)
        gdp_score = max(0.0, min(1.0, (self.gdp - 100) / 15.0))
        
        if self.task == "easy":
            return round(gdp_score, 2)
            
        unemp_score = max(0.0, min(1.0, 1.0 - (self.unemployment - 3.0) / 7.0))
        if self.task == "medium":
            return round((gdp_score + unemp_score) / 2.0, 2)
            
        ineq_score = max(0.0, min(1.0, 1.0 - (self.inequality - 0.2) / 0.6))
        return round((gdp_score + unemp_score + ineq_score) / 3.0, 2)