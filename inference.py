import os
import re
from openai import OpenAI
from environment import TaxPolicyEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1/")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct") 
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def run_baseline(task_name: str):
    env = TaxPolicyEnv(task=task_name)
    obs = env.reset()
    
    print(f"[START] task={task_name} env=tax_policy_env model={MODEL_NAME}", flush=True)
    
    done = False
    rewards_history = []
    
    while not done:
        prompt = (
            f"You are a tax policy AI. The current state is: GDP={obs.gdp}, "
            f"Tax={obs.tax_rate}%, Unemployment={obs.unemployment}%, Inequality={obs.inequality}. "
            f"Task: {task_name}. Respond ONLY with a number between -5.0 and 5.0 representing the tax rate change."
        )
        
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10
            )
            raw_text = response.choices[0].message.content.strip()
            match = re.search(r'-?\d+(\.\d+)?', raw_text)
            action_val = float(match.group()) if match else 0.0
            error_msg = "null"
        except Exception as e:
            action_val = 0.0
            error_msg = f"'{str(e)}'"
            
        action = Action(tax_change=action_val)
        obs, reward_obj, done, info = env.step(action)
        
        reward = reward_obj.value
        rewards_history.append(reward)
        
        print(f"[STEP] step={obs.step_count} action={action_val} reward={reward:.2f} done={str(done).lower()} error={error_msg}", flush=True)
    
    # Calculate final score natively in [0, 1] as required
    score = info.get('grader_score', 0.0)
    success = score >= 0.5
    rewards_str = ",".join([f"{r:.2f}" for r in rewards_history])
    
    # NEW FORMAT: Added score={score:.3f} to comply with the latest sample
    print(f"[END] success={str(success).lower()} steps={obs.step_count} score={score:.3f} rewards={rewards_str}", flush=True)

if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_baseline(task)