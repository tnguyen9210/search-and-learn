#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM

from .utils import Beam, build_conv, generate_k_steps, last

logger = logging.getLogger()
from sal.utils.score import aggregate_scores


'''
Config: 
* chat template 
- system_prompt: 
        "Solve the following math problem efficiently and clearly:\n\n- 
         For simple problems (2 steps or fewer):\nProvide a concise solution with 
         minimal explanation.\n\n- 
         For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## 
         Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## 
         Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\n
         Regardless of the approach, always conclude with:\n\n
         Therefore, the final answer is: $\\boxed{answer}$. 
         I hope it is correct.\n\nWhere [answer] is just the final number 
         or expression that solves the problem."

- custom_chat_template: 
        '{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'
        A Jinja2 template defining the structure of model interactions:
            - Structure Conversations: It organizes messages from different roles 
                (e.g., system, user, assistant, tool) into a specific layout using 
                headers like <|start_header_id|>role<|end_header_id|>.

            - Handle Tools: It Provides instructions for calling tools and formats 
                tool calls and responses in JSON, either in the system or user 
                section based on configuration.

            - Provide Metadata: It includes metadata such as the current date, 
                environment (e.g., IPython), and a "cutting knowledge date" to 
                give the model contextual awareness.

            - Ensure Compatibility: It enforces specific formats for tool calls 
                and responses, ensuring the model adheres to expected 
                input/output structures.
        
        General structure of output
            <|start_header_id|>system<|end_header_id>

            [Environment: ipython (if tools defined)]
            Cutting Knowledge Date: December 2023
            Today Date: [date_string]

            [tool instructions and list (if tools defined and tools_in_user_message is false)]
            [system_message]<|eot_id>

            [Optional: <|start_header_id|>user<|end_header_id>

            [tool instructions and list (if tools_in_user_message is true)]
            [first_user_message]<|eot_id>]

            [Formatted remaining messages based on role and content]

            [Optional: <|start_header_id|>assistant<|end_header_id> (if add_generation_prompt is true)]
        
        Example:
            Input:
                system_message: "Solve math problems step-by-step."
                messages: [{"role": "user", "content": "What is 2 + 2?"}]
                tools: [{"name": "add", "parameters": {"a": "int", "b": "int"}}]
                tools_in_user_message: true
                add_generation_prompt: true

            Output: 
                <|start_header_id|>system<|end_header_id>
                Environment: ipython
                Cutting Knowledge Date: December 2023
                Today Date: 26 Jul 2024

                Solve math problems step-by-step.
                <|eot_id>
                <|start_header_id|>user<|end_header_id>
                Given the following functions, please respond with a JSON for a 
                function call with its proper arguments that best answers the 
                given prompt.
                Respond in the following format:
                {"name": function name, 
                 "parameters": dictionary of argument name and its value}
                Available functions:
                [{"name": "add", "parameters": {"a": "int", "b": "int"}}]
                What is 2 + 2?
                <|eot_id>
                <|start_header_id|>assistant<|end_header_id>

* search parameters 
- n: int = 4: Number of samples (for "best_of_n") or beams (for search methods)
- temperature: float = 0.8: Controls randomness in sampling (higher = more random).
- top_p: float = 1.0: Filters sampling to the top probability mass (1.0 = full distribution).
- prm_batch_size: int = 4: Batch size for the Process Reward Model.
- search_batch_size: int = 25: Batch size for the search process (except for beam search).
- max_tokens: int = 2048: Maximum tokens to generate per output.
- agg_strategy: str = "last": Aggregation strategy for scores or selections 
    (options: "last", "min", "prod")
    last: The overall sequence score is the last step-wise score provided by the PRM.
    prod: The overall sequence score is the product of all step-wise scores provided by the PRM.
'''

'''
SamplingParams: sampling parameters for text generation 
https://github.com/vllm-project/vllm/blob/550d97eb58f03b21f0f4c9ef1935c2789186a5a0/vllm/sampling_params.py#L88
- n: Number of output sequences to *return* for the given prompt.
- best_of: Number of output sequences that are generated from the prompt.
    From these `best_of` sequences, the top `n` sequences are returned.
    `best_of` must be greater than or equal to `n`. By default,
    `best_of` is set to `n`.
- temperature: It controls the randomness of the sampling, lower more deterministic. 
    zero means greedy sampling
- top_p: float = 1.0: Filters sampling to the top probability mass (1.0 = full distribution).
- stop: List of strings that stop the generation when they are generated.
    The returned output will not contain the stop strings.
'''

def _beam_search(batch_of_prompts, config: Config, llm: LLM, prm: PRM) -> list[Beam]:
    '''
    Core beam search algorithm that generate n sequences
    * Args
    - batch_of_prompts: 
        a list of input prompts ["What is the capital of USA", "What is the length of the Nile River?"]
        config: a Config object with settings
    '''

    sampling_params = SamplingParams(temperature=config.temperature,
                                    max_tokens=config.max_tokens,
                                    top_p=config.top_p,
                                    stop=["\n\n"],
                                    include_stop_str_in_output=True,
                                    n=1)

    # ipdb.set_trace()
    beams: list[Beam] = []
    # create config.n beams for each prompt in the batch.
    for prompt in batch_of_prompts:
        for i in range(config.n):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    pruned=False,           # flag to track whether beam is active
                    completed=False,        # flag to track completion
                    stop_reasons=None,
                    history=[],             # list of generated text segments
                    best_scores=[],
                    all_scores=[],
                    previous_text=None,
                    completion_tokens=0,    # token counts
                )
            )

    completed_beams: list[Beam] = []

    for i in range(config.num_iterations):
        # filters beams to get only active (non-pruned) ones
        # first iteration i==0: uses all intial beams 
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        '''
        how the algorithm works:
        - 1. at iteration i = 0, we have n active beams 
        - 2. for each active beams we generate one next step (node)
        - 3. we selects the top s = n/beam-width beams to be active beams for next iteration
        - 4. at iteration i = 1, we duplicate the s active beams by beam-width
        - 5. continue step 2. 
        '''
        # duplicate active beams to ensure that we have config.n beams per iteration
        if len(active_beams) != config.n:
            repeats = (config.n // len(active_beams)) + 1
            logger.debug(
                f"Extending active_beams with {repeats} repetitions to reach size {config.n}"
            )
            extended_active_beams = [
                copy.deepcopy(b) for b in (active_beams * repeats)[: config.n]
            ]
            active_beams = extended_active_beams
            if len(active_beams) != config.n:
                raise ValueError(
                    f"Expected {config.n} active beams, but got {len(active_beams)}"
                )

        # adjust sampling_params for the last iterations
        # one difference from the general sampling_params
        # remove the parameter stop strings
        if i == config.num_iterations - 1:
            # Last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=1,
            )

        # build conversation dictionary that will be used later in apply_chat_template
        # this specifies three roles and their contents
        # system (system_promt), user (prompt), assistant (chatgpt response)
        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active_beams
        ]

        # set flags for conversation formatting
        # continue_final_message: True after the first iteration to append to
        # existing text.
        # add this text '[Continue with the solution steps...]'
        continue_final_message = i > 0

        # add_generation_prompt: True only on the first iteration to
        # start generation.
        # add this text '<|start_header_id|>assistant<|end_header_id>' 
        add_generation_prompt = i == 0      

        # override the default chat template if a custom one is provided.
        tokenizer = llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template

        '''
        apply_chat_template: format conversations using the chat template.
        # https://huggingface.co/docs/transformers/main/en/chat_templating

        *** i == 0 ***
        <|start_header_id|>system<|end_header_id>

        Cutting Knowledge Date: December 2023
        Today Date: 26 Feb 2025

        Solve the following math problem efficiently and clearly:

        - For simple problems (2 steps or fewer):
        Provide a concise solution with minimal explanation.

        - For complex problems (3 steps or more):
        Use this step-by-step format:

        ## Step 1: [Concise description]
        [Brief explanation and calculations]

        ## Step 2: [Concise description]
        [Brief explanation and calculations]

        ...

        Regardless of the approach, always conclude with:

        Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

        Where [answer] is just the final number or expression that solves the problem.
        <|eot_id|>

        <|start_header_id|>user<|end_header_id>

        Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.  
        Enter your answer in the form $(r,\\theta),$ where $r > 0$ and $0 \\le \\theta < 2 \\pi.$
        <|eot_id|>

        <|start_header_id|>assistant<|end_header_id>

        *** i == 1 ***

        <|start_header_id|>system<|end_header_id>

        Cutting Knowledge Date: December 2023
        Today Date: 26 Feb 2025

        Solve the following math problem efficiently and clearly:

        - For simple problems (2 steps or fewer):
        Provide a concise solution with minimal explanation.

        - For complex problems (3 steps or more):
        Use this step-by-step format:

        ## Step 1: [Concise description]
        [Brief explanation and calculations]

        ## Step 2: [Concise description]
        [Brief explanation and calculations]

        ...

        Regardless of the approach, always conclude with:

        Therefore, the final answer is: $\\boxed{answer}$. I hope it is correct.

        Where [answer] is just the final number or expression that solves the problem.
        <|eot_id|>

        <|start_header_id|>user<|end_header_id>

        Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.
        Enter your answer in the form $(r,\\theta)$, where $r > 0$ and $0 \\le \\theta < 2 \\pi$.
        <|eot_id|>

        <|start_header_id|>assistant<|end_header_id>

        ## Step 1: Recall the conversion formulas from rectangular to polar coordinates.
        The formulas are $r = \\sqrt{x^2 + y^2}$ and $\\theta = \\arctan\\left(\\frac{y}{x}\\right)$ for a point $(x, y)$.

        [Continue with the solution steps...] 
        
        Notes:
        <|eot_id|> indicates the end of a message or section within a structured 
            chat template used for language model interactions.
        '''
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=True       # return strings, not token IDs, e.g.,
                                # [[128006, 9125, 128007, 271, 38766]?
        )

        # set number of lookahead steps, 0 (no lookahead) on the last iteration 
        lookahead = 0 if i == config.num_iterations - 1 else config.lookahead

        '''
        # generate_k_steps: generate Beam objects with next (immediate) step
        # and lookahead steps.
        # If there are 6 templated_convs (prompts) then, it will generate
        # 6 Beam objects, each beam have 1 next step and 1 lookahead steps
        # the 
        # Notes
        # 1. beam width is already created when 
        # 2. next step is a sequence that end with double newlines "\n\n"
            e.g., ## Step 1: .... \n\n
            typically next step only consists of one thinking step
            i.e., ## Step 1: ... \n\n
            however, there are cases where it consists multiple thinking steps
            i.e., ## Step 1: ... \n ## Step 2: ... \n ## Step 3: ... \n\n 
        '''
        gen_results = generate_k_steps(
            templated_convs,    # list of templated conversion (prompts)
            lookahead,
            llm,
            sampling_params,
            1                   # beam_width
        )

        prompts, completions = [], []

        # update each beam with generation results 
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts  # next immediate steps
            beam.stop_reasons = gen_result.stop_reasons  # stop reasons
            beam.lookahead_texts = gen_result.lookahead_texts  # 
            beam.completion_tokens += gen_result.completion_tokens  # tokens count
            beam.current_text += beam.next_texts[0]                 # current text sequence until now (only solution steps, exclude initial prompt)
            beam.history.append(beam.next_texts[0])

            # mark a beam as completed if generation ends
            # generation ends required beam satisfying one of 3 criteria
            if (
                beam.stop_reasons[0] == "EOS"       # stop reason is end of sequence
                or beam.stop_reasons[0] == "length"  # not sure when 
                or beam.next_texts[0] == ""          # llm can not generate next steps 
            ):
                beam.completed = True
                completed_beams.append(beam)

            # collect prompts and completions for scoring
            prompts.append(beam.prompt)
            completions.append([beam.current_text])

        # compute scores and aggregate scores for each completion
        # three aggregate methods: last, min, product
        # not sure how the PRM works. for a completion with outputs 3 scores for a completion with 
        '''
        completions
        [['## Step 1: Understand the conversion formulas\nTo convert from rectangular (Cartesian) coordinates $(x, y)$ to polar coordinates $(r, \\theta)$, we use the following formulas:\n$r = \\sqrt{x^2 + y^2}$ for the radial coordinate,\n$\\theta = \\arctan\\left(\\frac{y}{x}\\right)$ for the angular coordinate.\n\n## Step 2: Calculate the radial coordinate $r$\nUsing the formula $r = \\sqrt{x^2 + y^2}$, we substitute $x = 0$ and $y = 3$:\n$r = \\sqrt{0^2 + 3^2} = \\sqrt{0 + 9} = \\sqrt{9} = 3$.\n\n'], ['## Step 1: Recall the formulas for converting rectangular coordinates to polar coordinates\nThe relationship between rectangular coordinates $(x,y)$ and polar coordinates $(r,heta)$ is given by $x = r\\cos(heta)$ and $y = r\\sin(heta)$.\n\n## Step 2: Plug in the given rectangular coordinates into the formulas\nWe have $(x,y) = (0,3)$, so $x = 0 = r\\cos(heta)$ and $y = 3 = r\\sin(heta)$.\n\n']]
        scores
        [[[0.99609375, 1.0, 0.9609375]], [[0.99609375, 1.0, 0.96875]]]
        [[0.9609375], [0.96875]]
        '''
        scores = prm.score(prompts, completions)
        agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in scores
        ]

        for beam, score in zip(active_beams, scores, strict=True):
            beam.all_scores = score[0]


        # filter active_beams and agg_scores for beams that are completed
        agg_scores = [
            agg_scores[i] for i, b in enumerate(active_beams) if not b.completed
        ]
        active_beams = [b for b in active_beams if not b.completed]

        # early stopping if all beams are completed
        if len(active_beams) == 0:
            break

        # filter duplicate active beams
        if config.filter_duplicates:
            # create a dictionary to filter duplicates and retain order
            unique_beam_dict = {}
            for i, b in enumerate(active_beams):
                if b.current_text not in unique_beam_dict:
                    unique_beam_dict[b.current_text] = (
                        i  # Map the unique text to its index
                    )
            active_beams = [active_beams[i] for i in unique_beam_dict.values()]
            agg_scores = [agg_scores[i] for i in unique_beam_dict.values()]

        # get indices for top (config.n / config.beam_width) completions
        top_indices = np.argsort(np.array(agg_scores).flatten())[
            -(config.n // config.beam_width) :
        ]

        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True

    # # filter completed beams for those with top config.n scores
    # if config.sort_completed:
    #     completed_beams = sorted(
    #         completed_beams,
    #         key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
    #         reverse=True,
    #     )[: config.n]
    # else:
    #     completed_beams = completed_beams[: config.n]

    if len(completed_beams) != config.n:
        # if we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (config.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {config.n}"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    return completed_beams


def beam_search(examples, config: Config, llm: LLM, prm: PRM):
    '''
    Algorithm that uses _beam_search to generate n sequence, then
        picks the sequence with highest aggregate score 
    '''
    problems = examples["problem"]
    # generate n solution sequences 
    beam_results = _beam_search(problems, config, llm, prm)

    # group together alike beams and store in the dataset
    grouped_results = defaultdict(list)
    for results in beam_results:
        grouped_results[results.prompt].append(results)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    # for each of the problem, find the best 
    # 
    for p in problems:
        beams = grouped_results[p]
        completions = [b.current_text for b in beams]
        agg_scores = [
            aggregate_scores(b.all_scores, config.agg_strategy) for b in beams
        ]
        pred = completions[np.argmax(agg_scores)]
        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in beams])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in beams])

    return results
