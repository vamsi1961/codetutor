import os
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.tools import PythonREPLTool
from langgraph.prebuilt import create_react_agent as create_graph_react_agent
from langchain.agents import create_react_agent as create_chain_react_agent
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import operator
from typing import Annotated, List, Tuple, Dict, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import Union
from langchain.tools import Tool
from langgraph.graph import END
from langgraph.graph import StateGraph, START
from langchain.agents import AgentExecutor
from langchain import hub
import subprocess
import re
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.prompts import PromptTemplate
from langchain.tools.render import render_text_description
from langchain.schema import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException


# Load environment variables
load_dotenv()

# Configure Azure OpenAI
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
gpt4_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")

# Disable LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Initialize Azure OpenAI LLM
llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    deployment_name=gpt4_deployment_name,
    api_version="2024-08-01-preview",
    temperature=0
)

def clean_code_string(code: str) -> str:
    # Strip surrounding triple quotes if present

    if code.startswith("'''") and code.endswith("'''"):
        code = code[3:-3]
    
    # Handle markdown code blocks
    if code.startswith("```python") and code.endswith("```"):
        code = code[len("```python"):-3].strip()
    elif code.startswith("```") and code.endswith("```"):
        code = code[3:-3].strip()
    
    # Ensure no language identifier at start of file
    code_lines = code.strip().split('\n')
    if code_lines and code_lines[0].strip() == 'python':
        code = '\n'.join(code_lines[1:])
    
    # Optionally strip a trailing quote if one got appended wrongly
    code = code.rstrip('"\'')
    
    return code.strip()

def write_execute_py(code: str) -> str:
    try:
        # Clean the code string
        code = clean_code_string(code)
        
        print(f"Writing to machine.py:\n{code}")
        
        with open("machine.py", "w") as f:
            f.write(code)
            
        print("Code written to machine.py, now executing...")
        
        # Run the file
        result = subprocess.run(["python", "machine.py"], capture_output=True, text=True, timeout=10)
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        
        if result.returncode == 0:
            return f"APPROVED - machine.py ran successfully.\nOutput:\n{result.stdout.strip()}"
        else:
            return f"NEEDS REVISION - machine.py failed.\nErrors:\n{result.stderr.strip()}"

    except Exception as e:
        return f"Failed to update machine.py: {str(e)}"
    except subprocess.TimeoutExpired:
        return "NEEDS REVISION - machine.py timed out during execution."

# Create a direct file writing function (without execution)
def write_code_to_file(code: str) -> str:
    """Simply writes code to machine.py without executing it."""
    try:
        code = clean_code_string(code)
        
        print(f"Writing to machine.py (without execution):\n{code}")
        
        with open("machine.py", "w") as f:
            f.write(code)
            
        return f"Code written to machine.py successfully."
    except Exception as e:
        return f"Failed to write to machine.py: {str(e)}"


# Extract code from text that might contain Python code
def extract_code_from_text(text: str) -> Optional[str]:
    """Extract Python code from text that might contain code blocks or raw code."""
    # Try to extract code blocks first
    code_match = re.search(r'```python\s*([\s\S]*?)\s*```', text)
    if code_match:
        return code_match.group(1).strip()
    
    # Try to extract code blocks without language specifier
    code_match = re.search(r'```\s*([\s\S]*?)\s*```', text)
    if code_match:
        return code_match.group(1).strip()
    
    # If no code blocks, check if the text itself looks like code
    if "def " in text or "import " in text or "print(" in text:
        return text.strip()
    
    return None

# Create a function to read existing code
def read_existing_code() -> str:
    """Read the content of machine.py if it exists."""
    try:
        if os.path.exists("machine.py"):
            with open("machine.py", "r") as f:
                return f.read()
        return "No existing code found."
    except Exception as e:
        return f"Error reading machine.py: {str(e)}"

# Global variable to store completed tasks and their observations
completed_tasks = []

def writer_agent_handler(input_text):
    """Custom handler for WriterAgent that ensures code is saved properly and includes task history."""
    try:
        # Get the current code
        existing_code = read_existing_code()
        
        # Generate context from completed tasks
        tasks_context = ""
        if completed_tasks:
            tasks_context = "Previously completed tasks:\n"
            for idx, (task, observation) in enumerate(completed_tasks):
                tasks_context += f"Task {idx+1}: {task}\nResult: {observation}\n\n"
        
        # Create enhanced input with code context and task history
        enhanced_input = f"""
Current requirement: {input_text}

{tasks_context}
        """
        
        print(f"Calling WriterAgent with context of {len(completed_tasks)} previous tasks")
        result = writer_executor.invoke({"input": enhanced_input})
        
        # Extract code if the result contains Python code block
        output = result.get("output", "")
        code_match = re.search(r'```python\s*([\s\S]*?)\s*```', output)
        
        if code_match:
            # We found some code in the output
            code = code_match.group(1)
            print("Found code in WriterAgent output, saving to file...")
            write_code_to_file(code)
            
        return output
    except Exception as e:
        print(f"Error in WriterAgent: {e}")
        return f"Error in WriterAgent: {e}"

# Custom ReAct parser with improved error handling
class CustomReActSingleInputOutputParser(ReActSingleInputOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            return super().parse(text)
        except OutputParserException as e:
            # Check if it contains both an action and a final answer
            if "both a final answer and a parse-able action" in str(e):
                # Prioritize the Action over Final Answer
                if "Action:" in text and "Action Input:" in text:
                    # Extract just the action parts
                    action_parts = text.split("Action:")
                    if len(action_parts) > 1:
                        action_text = "Action:" + action_parts[1].split("Thought: I now know the final answer")[0]
                        return super().parse(action_text)
                # If no action found, try to extract the final answer
                if "Final Answer:" in text:
                    final_answer_parts = text.split("Final Answer:")
                    if len(final_answer_parts) > 1:
                        final_answer = final_answer_parts[1].strip()
                        return AgentFinish({"output": final_answer}, text)
            
            # Re-raise if we couldn't handle it
            raise e

# Create tools
FileWriteTool = Tool(
    name="WriteToFile",
    func=write_execute_py,
    description="Writes the provided Python code to machine.py file and executes it if it says it require revision then you have to call PythonREPLTool to re-write the code. Input should be valid Python code as a string."
)

# Get the existing code initially
existing_code = read_existing_code()
print(f"Existing code found: {'Yes' if existing_code != 'No existing code found.' else 'No'}")

# Function to generate instructions with updated task history
def generate_writer_instructions(tasks_history=None):
    existing_code = read_existing_code()
    tasks_context = ""
    
    if tasks_history and len(tasks_history) > 0:
        tasks_context = "\nPreviously completed tasks and their results:\n"
        for idx, (task, observation) in enumerate(tasks_history):
            tasks_context += f"Task {idx+1}: {task}\nResult: {observation}\n\n"
    
    instructions = f"""
    Think as if you are teaching code to a beginner so you should not change the previous generated code just modify it according to the new task
    You are an agent designed to write Python code based on user requirements.
    
    Here is the EXISTING CODE from machine.py that you should use as your starting point:
    
    ```python
    {existing_code}
    ```
    {tasks_context}
    If you get any error update code accordingly
    After writing the code dont execute at all.
    See what you can add to meet the requirements if you have to remove it to meet tasks_context then remove it
    Modify the existing code to meet the new requirements rather than writing from scratch.
    Only make necessary changes to fulfill the requirements.
    just write code for tasks_context dont think much.
    
    Do NOT include ```python or ``` markers in your final answer - write only valid raw Python code.
    """
    return instructions

# Code Writer Agent with embedded existing code and task history
writer_prompt = hub.pull("langchain-ai/react-agent-template").partial(
    instructions=generate_writer_instructions()
)

# Initial writer agent setup
writer_agent = create_chain_react_agent(llm=llm, tools=[PythonREPLTool(), FileWriteTool], prompt=writer_prompt)
writer_executor = AgentExecutor(agent=writer_agent, tools=[PythonREPLTool(), FileWriteTool], verbose=True, handle_parsing_errors=True)

# Create the WriterTool with enhanced context
WriterTool = Tool(
    name="WriterAgent",
    func=writer_agent_handler,
    description="Writes or updates machine.py with Python code based on the input prompt. Automatically extracts and saves any code blocks in the response. This tool has access to the history of previously completed tasks."
)

# Evaluator agent and tool setup
evaluator_prompt = hub.pull("langchain-ai/react-agent-template").partial(
    instructions="""You are an agent that checks if the code in machine.py works.
    Use the 'RunMachinePy' tool to run it and evaluate its correctness.
    If the code runs without errors, return 'APPROVED'.
    If not, return 'NEEDS REVISION' and describe the problem.
    """
)

evaluator_agent = create_chain_react_agent(llm=llm, tools=[PythonREPLTool(), FileWriteTool], prompt=evaluator_prompt)
evaluator_executor = AgentExecutor(agent=evaluator_agent, tools=[PythonREPLTool(), FileWriteTool], verbose=True, handle_parsing_errors=True)

EvaluationTool = Tool(
    name="EvaluationAgent",
    func=lambda input_text: evaluator_executor.invoke({"input": input_text}),
    description="Execute the function and pass the message"
)

tools = [WriterTool, EvaluationTool]

# Define the React agent template with STRICT FORMAT instructions
template = """
    Answer the following questions as best you can. You have access to the following tools:
    {tools}
    
    Use the following format STRICTLY:
    
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    
    Thought: Unless and until the EvaluationTool says code is APPROVED then only you have to confirm that code is good and end the agent
    After wrtiting the code using WriterTool you have to use EvaluationTool to check if it correct or not
    Final Answer: the final answer to the original input question
    
    Begin!
    
    Question: {input}
    {agent_scratchpad}
    """

prompt = PromptTemplate.from_template(template=template).partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# Create the agent with the modified parser
code_agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
    }
    | prompt
    | llm
    | CustomReActSingleInputOutputParser()
)

# Define state types for the workflow
class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

class Response(BaseModel):
    """Response to user."""
    response: str

class Act(BaseModel):
    """Action to perform."""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

# Define prompts for planning and replanning
planner_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """For the given objective, create a step-by-step plan on how to write the code just give a direct task no drama.
        try to get it in 2-4 steps dont take more steps
        Your steps will be executed by a tool that can write and run code.
        Write concise, clear steps that explain what needs to be done.
        Focus only on the necessary steps to complete the objective.
        """,
    ),
    ("placeholder", "{messages}"),
])

replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, update your step-by-step plan based on progress so far.

Your objective was this: {input}

Your original plan was: {plan}

You have currently completed these steps: {past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, 
provide a final response. Otherwise, list only the remaining steps that need to be done.
Do not include previously completed steps in your updated plan."""
)

# Create the planner and replanner
planner = planner_prompt | llm.with_structured_output(Plan)
replanner = replanner_prompt | llm.with_structured_output(Act)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

# Define workflow functions
def plan_step(state: PlanExecute):
    plan = planner.invoke({"messages": [("user", state["input"])]})
    return {"plan": plan.steps}

def replan_step(state: PlanExecute):
    output = replanner.invoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        print(" I am replanning")
        return {"plan": output.action.steps}

def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"

def run_agent_with_steps(agent, tools, input_text: str, completed_tasks, max_iterations: int = 10):
    """Run the agent for multiple iterations until it reaches a final answer or max iterations"""
    intermediate_steps = []
    code_accumulated = None
    
    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1} ---")
        try:
            agent_step = agent.invoke(
                {
                    "input": input_text,
                    "agent_scratchpad": intermediate_steps,
                }
            )
            
            print(f"Agent output: {agent_step}")
            
            if isinstance(agent_step, AgentFinish):
                print("### Agent Finished ###")
                final_output = agent_step.return_values['output']
                print(f"Final Answer: {final_output}")
                
                # Try to extract code from the final answer and save it if found
                code = extract_code_from_text(final_output)
                if code:
                    print("Found code in final answer, saving to file...")
                    write_code_to_file(code)
                elif code_accumulated:
                    print("Using accumulated code from previous steps...")
                    write_code_to_file(code_accumulated)
                    
                return agent_step
                
            if isinstance(agent_step, AgentAction):
                tool_name = agent_step.tool
                print(f"Selected tool: {tool_name}")
                tool_to_use = find_tool_by_name(tools, tool_name)
                tool_input = agent_step.tool_input
                
                # If it's the WriterAgent, update its instructions with task history
                if tool_name == "WriterAgent":
                    # Update the writer agent with the latest task history before calling
                    updated_instructions = generate_writer_instructions(completed_tasks)
                    writer_prompt = hub.pull("langchain-ai/react-agent-template").partial(
                        instructions=updated_instructions
                    )
                    writer_agent = create_chain_react_agent(llm=llm, tools=[PythonREPLTool(), FileWriteTool], prompt=writer_prompt)
                    writer_executor = AgentExecutor(agent=writer_agent, tools=[PythonREPLTool(), FileWriteTool], verbose=True, handle_parsing_errors=True)
                
                # Try to extract code from the tool input
                code = extract_code_from_text(tool_input)
                if code and (tool_name == "WriterAgent" or tool_name == "WriteToFile" or tool_name == "DirectWriteToFile"):
                    code_accumulated = code
                
                observation = tool_to_use.func(str(tool_input))
                print(f"Observation: {observation}")
                intermediate_steps.append((agent_step, str(observation)))
                print(f"Intermediate steps updated")
        
        except OutputParserException as e:
            print(f"Parsing error: {e}")
    
            # If we have code accumulated but couldn't recover normally, write it to file
            if code_accumulated:
                print("Writing accumulated code to file before failing...")
                write_code_to_file(code_accumulated)
            
            # If we couldn't recover, we'll treat this as a final step
            print("Could not recover from parsing error, treating as final step")
            return AgentFinish({"output": f"Error in agent execution, but code has been saved to machine.py. Error details: {e}"}, "")
            
    print("Reached maximum iterations without finishing")
    
    # If we have code accumulated but reached max iterations, write it to file
    if code_accumulated:
        print("Writing accumulated code to file before finishing...")
        write_code_to_file(code_accumulated)
        
    return AgentFinish({"output": "Reached maximum number of iterations, but the code has been saved to machine.py."}, "")

def execute_step(state: PlanExecute, max_iterations: int = 10) -> Dict[str, Any]:
    global completed_tasks
    
    if not state.get("plan"):
        print("No plan found in state")
        return {"response": "No steps to execute. Please provide valid instructions."}
    
    plan = state["plan"]
    if not plan:
        print("Empty plan")
        return {"response": "Empty execution plan. Please provide valid instructions."}
    
    # Create a formatted display of the plan
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    
    # Get the first step to execute
    task = plan[0]
    
    # Format the task with plan context
    task_formatted = f"""For the following plan: {plan_str}\n\n You are tasked with executing step {1}, {task}. """
    
    # Execute the agent with the current task and completed task history
    result = run_agent_with_steps(code_agent, tools, task_formatted, completed_tasks)
    
    # Handle the result
    if result is None:
        return {"past_steps": [(task, "Failed to complete")], "response": "Error executing the step, but any code written has been saved to machine.py."}
    
    if isinstance(result, AgentFinish):
        # Add the completed task to our history
        task_result = (task, result.return_values.get("output", "Task completed"))
        completed_tasks.append(task_result)
        return {"past_steps": [task_result]}
    
    # Fallback - should not reach here with the new implementation
    task_result = (task, "Task execution completed, code saved to machine.py.")
    completed_tasks.append(task_result)
    return {"past_steps": [task_result]}

# Build the workflow
workflow = StateGraph(PlanExecute)
workflow.add_node("planner", plan_step)
workflow.add_node("agent", execute_step)
workflow.add_node("replan", replan_step)

# Connect the nodes
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "agent")
workflow.add_edge("agent", "replan")
workflow.add_conditional_edges(
    "replan",
    should_end,
    ["agent", END],
)

# Compile the workflow
app = workflow.compile()

# Define the input for the workflow
config = {"recursion_limit": 50}
inputs={"input": "write a code to add 2 numbers by taking input and logarithm of the result do step by step"}    

def main():
    # Reset the global completed_tasks before starting
    global completed_tasks
    completed_tasks = []
    
    print("Starting workflow to execute code generation task...")
    for event in app.stream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print("\n--- STEP UPDATE ---")
                print(v)
    print("\nWorkflow completed.")
    

# Execute the workflow
if __name__ == "__main__":
    main()