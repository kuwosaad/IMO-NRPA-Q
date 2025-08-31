"""
mcts.py

Monte Carlo Tree Search fallback for NRPA strategy selection.
Provides alternative search algorithm when NRPA fails to generate strategies.
"""

class MCTSSearch:
    def __init__(self, api_client_funcs, strategist_model_name):
        self.api_client_funcs = api_client_funcs
        self.strategist_model_name = strategist_model_name
    
    def run(self, problem_statement, other_prompts, system_prompt, telemetry):
        # Simplified MCTS implementation for strategy search
        strategist_payload = self.api_client_funcs["build_request_payload"](
            system_prompt=system_prompt,
            question_prompt=problem_statement,
            other_prompts=other_prompts,
        )
        strategist_response = self.api_client_funcs["send_api_request"](
            self.api_client_funcs["get_api_key"]("strategist"),
            strategist_payload,
            model_name=self.strategist_model_name,
            agent_type="strategist",
            telemetry=telemetry,
        )
        return self.api_client_funcs["extract_text_from_response"](strategist_response)

def run_mcts_search(problem_statement, other_prompts, system_prompt, api_client_funcs, strategist_model_name, telemetry):
    return MCTSSearch(api_client_funcs, strategist_model_name).run(
        problem_statement, other_prompts, system_prompt, telemetry
    )