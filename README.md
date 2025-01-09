# multi-agent-RAG.ipynb
Framework Using LangGraph

![image](https://github.com/user-attachments/assets/722c9fba-b5ff-4f70-b3ff-b210644f7f7b)

```markdown
# Multi-Agent Collaboration Framework with RAG and LangChain

## Overview
This project implements a robust **multi-agent collaboration framework** using **LangChain** and **LangGraph**, designed to solve complex tasks through specialized agents. By combining **Retrieval-Augmented Generation (RAG)** and AI-driven decision-making, the framework enables seamless task delegation and execution across multiple agents.

## Key Features
- **Dynamic Multi-Agent Collaboration**: Specialized agents handle research, chart generation, and tool execution tasks.
- **RAG Integration**: Retrieves and processes data dynamically using advanced AI techniques.
- **Secure and Scalable**: Sensitive API keys are securely handled, and the modular design allows easy scalability.
- **State Graph Workflow**: Implements a conditional routing system for efficient agent collaboration.
- **Real-World Examples**: Pre-built examples include tasks like GDP analysis, climate change visualization, and renewable energy adoption trends.

## Installation
To run this project, install the required dependencies:
```bash
pip install -U langchain langchain_openai langsmith pandas langchain_experimental matplotlib langgraph langchain_core
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/multi-agent-collaboration.git
   cd multi-agent-collaboration
   ```

2. Set up your environment variables for API keys securely:
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export LANGCHAIN_API_KEY="your-langchain-api-key"
   export TAVILY_API_KEY="your-tavily-api-key"
   ```

3. Run the script:
   ```bash
   python multi_agent_collaboration.py
   ```

4. Explore the pre-built examples or add your custom tasks.

## Examples
The framework demonstrates the following use cases:
1. **GDP Analysis**: Fetches the UK's GDP over the past 5 years and generates a line graph.
2. **Climate Change Visualization**: Summarizes climate change statistics for the last decade and creates visualizations.
3. **Renewable Energy Trends**: Analyzes global renewable energy adoption rates and generates a bar chart.

Output events for each task are streamed in real-time, showcasing the agent collaboration process.

## Project Architecture
- **Agents**:
  - **Research Agent**: Uses the Tavily search tool to retrieve data.
  - **Chart Generator Agent**: Executes Python code to create visualizations.
- **Tools**:
  - **Tavily Search Tool**: Retrieves web-based search results.
  - **Python REPL Tool**: Executes Python code dynamically.
- **State Graph**: Coordinates the workflow using LangGraph, routing tasks between agents and tools.
- **RAG Framework**: Dynamically retrieves relevant information to enhance task execution.

## Contributing
Contributions are welcome! If you have ideas to improve this framework or new use cases to add, feel free to submit a pull request.

## Acknowledgments
This project is inspired by the paper [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155) and uses the [LangGraph](https://github.com/langchain-ai/langgraph) library.

---

For further questions, feel free to open an issue or contact us!
```
@Abs
