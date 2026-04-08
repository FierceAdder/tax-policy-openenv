import sys
import os
import uvicorn
from fastapi import FastAPI

# Allow this script to import environment.py from the root folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import TaxPolicyEnv, Action

app = FastAPI()
env = TaxPolicyEnv(task="easy")

@app.post("/reset")
def reset():
    obs = env.reset()
    return {"observation": obs}

@app.post("/state")
def state():
    obs = env.state()
    return {"observation": obs}

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/")
def read_root():
    return {"status": "Tax Policy Env is running"}

# The entry point the validator is looking for!
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()