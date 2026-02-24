from typing import Dict, List, Any
import random

class Message:
    def __init__(self, sender_id: str, recipient_id: str, payload: Any):
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.payload = payload


class Agent:
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.state = {
            "last_observed_total": 0,
            "proposed_increment": 1
        }
        self.inbox: List[Message] = []

    def receive(self, message: Message):
        self.inbox.append(message)

    def observe(self, observation: Dict[str, Any]):
        self.state["last_observed_total"] = observation["total"]

    def policy(self) -> Dict[str, Any]:
        total = self.state["last_observed_total"]
        increment = self.state["proposed_increment"]

        if total >= 20:
            increment = 0
        else:
            increment = random.choice([1, 2])

        self.state["proposed_increment"] = increment
        return {"increment": increment}

    def clear_inbox(self):
        self.inbox.clear()


class Environment:
    def __init__(self, agents: List[Agent]):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.total = 0
        self.time_step = 0

    def broadcast(self, sender_id: str, payload: Any):
        for agent_id, agent in self.agents.items():
            if agent_id != sender_id:
                message = Message(sender_id, agent_id, payload)
                agent.receive(message)

    def step(self):
        self.time_step += 1

        # Observation phase (consistent snapshot)
        observation = {"total": self.total}
        for agent in self.agents.values():
            agent.observe(observation)

        # Action phase (ordered collection)
        proposed_actions = {}
        for agent_id, agent in self.agents.items():
            action = agent.policy()
            proposed_actions[agent_id] = action

        # Environment update (ordered fold)
        for action in proposed_actions.values():
            increment = action["increment"]
            if self.total + increment <= 20:
                self.total += increment

        # Communication phase (each agent rebroadcasts current total)
        for agent_id in self.agents:
            self.broadcast(agent_id, {"total": self.total})

        # Inbox lifecycle: messages are generated in this step and discarded
        # without influencing policy; they would be incorporated in a subsequent
        # step in a message-driven extension.
        for agent in self.agents.values():
            agent.clear_inbox()

    def run(self, max_steps: int = 50):
        while self.total < 20 and self.time_step < max_steps:
            self.step()

        return self.total


if __name__ == "__main__":
    agents = [Agent(f"agent_{i}") for i in range(3)]
    env = Environment(agents)
    final_total = env.run()
    print("Final total:", final_total)
