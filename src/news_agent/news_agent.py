from typing import Any, Dict, List, Literal

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from langgraph.types import Command, interrupt

# Import prompts for both news source and generic preference updates
from news_agent.prompts import (MEMORY_UPDATE_INSTRUCTIONS,
                                MEMORY_UPDATE_INSTRUCTIONS_NEWS_SOURCE)
# Structured output schemas for the two preference types
from news_agent.schemas import UserNewsSourcePreferences, UserPreferences
from news_agent.tools.tavily_tools import (tavily_crawl,
                                           tavily_extract_content,
                                           tavily_map_site, tavily_search)

default_news_source_preferences = """   

- TechCrunch
- The Verge
- The Wall Street Journal
- The New Yorker
- The Atlantic
- New York Times
- The Economist
- Associated Press
- Forbes
- Bloomberg
- The Economist
"""

default_content_preferences = """
- Technology and innovation news
- Business and finance developments
- AI and machine learning advancements
- Startup and venture capital news
- Digital transformation trends
- Economic policy and market analysis
- Media and journalism industry insights
"""


def get_memory(
    store,
    namespace,
    default_content,
):
    """Get memory from the store or initialize with default if it doesn't exist.

    Args:
        store: LangGraph BaseStore instance to search for existing memory

    Returns:
        str: The content of the memory profile, either from existing memory or the default
    """

    # Search for existing memory with namespace and key
    user_preferences = store.get(namespace, "user_preferences")

    if user_preferences:
        return user_preferences.value
    else:
        store.put(namespace, "user_preferences", default_content)

        return default_content


def update_memory(store, namespace, messages):
    """Update memory profile in the store.

    Args:
        store: LangGraph BaseStore instance to update memory
        namespace: Tuple defining the memory namespace, e.g. ("news_feed_agent", "news_source_preferences")
        messages: List of messages to update the memory with
    """

    # ----------------------------------------------------------------------------------
    # Determine which preference profile we are updating (news sources vs. content).
    # This allows us to keep the two preference types cleanly separated so that
    # websites do not bleed into content preferences and vice-versa.
    # ----------------------------------------------------------------------------------

    # The namespace is always of the form ("news_feed_agent", <preference_key>)
    preference_key = namespace[1] if len(namespace) > 1 else ""

    if preference_key == "news_source_preferences":
        instructions_prompt = (
            MEMORY_UPDATE_INSTRUCTIONS_NEWS_SOURCE
            + "\n\nIMPORTANT: The profile SHOULD ONLY list websites or publication names (e.g., 'TechCrunch', 'nytimes.com'). DO NOT include topics, themes, or content interests."
        )
        schema = UserNewsSourcePreferences
        preference_attribute_name = "user_news_source_preferences"
    elif preference_key == "content_preferences":
        # Dedicated instructions so that only content/topics of interest are captured
        instructions_prompt = """
# Role and Objective
You are a memory profile manager for a news feed agent that selectively updates the USER'S CONTENT PREFERENCES (topics, themes, areas of interest) based on feedback messages from human-in-the-loop interactions.

# Instructions
- NEVER overwrite the entire memory profile
- ONLY make targeted additions of new information
- ONLY update specific facts that are directly contradicted by feedback messages
- PRESERVE all other existing information in the profile
- The profile SHOULD ONLY talk about topics or areas of interest, NOT websites or sources.
- Format the profile consistently with the original style (bullet list).
- Generate the profile as a string

# Reasoning Steps
1. Analyse the current memory profile structure and content.
2. Review feedback messages from human-in-the-loop interactions.
3. Extract ONLY the content preferences (topics of interest) from these feedback messages.
4. Compare new information against existing profile.
5. Identify only specific facts to add or update.
6. Preserve all other existing information.
7. Output the complete updated profile.

# Example
<memory_profile>
- Very interested in Computer Vision and its manufacturing applications
- Interested in iOT and its applications in manufacturing
</memory_profile>

<user_messages>
"Please prioritise articles about sustainability and green manufacturing"
</user_messages>

<updated_profile>
- Very interested in Computer Vision and its manufacturing applications
- Interested in iOT and its applications in manufacturing
- Sustainability and green manufacturing
</updated_profile>

# Process current profile for {namespace}
<memory_profile>
{current_profile}
</memory_profile>

Think step by step about what specific feedback is being provided and what specific information should be added or updated in the profile while preserving everything else.

Think carefully and update the memory profile based upon these user messages:"""

        schema = UserPreferences
        preference_attribute_name = "user_preferences"
    else:  # Any unforeseen preference namespace falls back to generic behaviour
        instructions_prompt = MEMORY_UPDATE_INSTRUCTIONS
        schema = UserPreferences
        preference_attribute_name = "user_preferences"

    # Update the memory using the appropriate structured output schema so that the
    # resulting profile only contains the correct type of preference data.
    llm = init_chat_model("openai:gpt-4.1", temperature=0.0).with_structured_output(
        schema
    )

    # Get the existing memory (if this is the first time, fall back to an empty string)
    user_preferences_record = store.get(namespace, "user_preferences")
    existing_profile_value = (
        user_preferences_record.value if user_preferences_record else ""
    )

    # Update the memory
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            # Message already in the correct format
            formatted_messages.append(msg)
        elif hasattr(msg, "role") and hasattr(msg, "content"):
            # LangChain BaseMessage (e.g., AIMessage, HumanMessage)
            formatted_messages.append({"role": msg.role, "content": msg.content})
        else:
            # Fallback: convert to string and use as assistant content
            formatted_messages.append({"role": "assistant", "content": str(msg)})

    result = llm.invoke(
        [
            {
                "role": "system",
                "content": instructions_prompt.format(
                    current_profile=existing_profile_value, namespace=namespace
                ),
            },
        ]
        + formatted_messages
    )
    # Save the updated memory to the store
    updated_value: str
    if hasattr(result, preference_attribute_name):
        updated_value = getattr(result, preference_attribute_name)  # type: ignore[attr-defined]
    elif hasattr(result, "user_preferences"):
        updated_value = result.user_preferences  # type: ignore[attr-defined]
    elif hasattr(result, "user_news_source_preferences"):
        updated_value = result.user_news_source_preferences  # type: ignore[attr-defined]
    else:
        # Fallback: attempt to treat the result as a dict or string
        if isinstance(result, dict):
            updated_value = result.get("user_preferences") or result.get(
                "user_news_source_preferences", ""
            )
        else:
            updated_value = str(result)

    store.put(namespace, "user_preferences", updated_value)


class CrawlState(MessagesState):
    """State for the crawling agent."""

    crawl_results: List[Dict[str, Any]] = []
    discovered_urls: List[str] = []
    summary: str = ""


def crawl_agent(
    state: CrawlState, store: BaseStore
) -> Command[Literal["tools", "feedback"]]:
    """Intelligent crawling agent that can crawl websites, extract content, and search the web."""
    llm = init_chat_model("openai:gpt-4.1", temperature=0.0)
    llm_with_tools = llm.bind_tools(
        [tavily_crawl, tavily_map_site, tavily_search, tavily_extract_content]
    )

    # Get the news source preferences
    news_source_preferences = get_memory(
        store,
        ("news_feed_agent", "news_source_preferences"),
        default_news_source_preferences,
    )

    # Get the content preferences
    content_preferences = get_memory(
        store,
        ("news_feed_agent", "content_preferences"),
        default_content_preferences,
    )

    # Enhanced system prompt for crawling capabilities
    system_prompt = f"""You are an intelligent web news source aggregator. Your job is to provide curated news sources to the user.

There are three main modes you can operate in. Think about the user's message and determine which mode is most appropriate.
1. Daily News Debrief: Summarize the most important news from the user's preferred sources, focusing on their content preferences. The content you search should be timely so it should be some of the most recent news.
2. Fun Fact: Find and share an interesting, surprising, or quirky news fact from today's headlines from the user's preferred sources, matching their content preferences. Explain why the fact is interesting.
3. Single Source Summary: If the user requests a summary from a specific news source, only provide a summary from that source, filtered by their content preferences.

Based on the user's message, infer which mode is most appropriate and act accordingly. If the user message is ambiguous, default to the Daily News Debrief.
    
Available tools:
1. tavily_crawl: Crawl a website starting from a top-level page, extracting content from multiple pages
2. tavily_map_site: Map a website to discover all available URLs without extracting content
3. tavily_search: Search the web for specific information
4. tavily_extract_content: Extract content from specific URLs

If a tool you use does not return useful results (for example, the crawl returns no pages, an error, or irrelevant content), you should try a different tool or approach (such as using map, search, or extract_content) to answer the user's request.

Always provide a summary of what you found and suggest next steps if appropriate.

Do NOT ask any follow-up or clarifying questions. Treat the user's message as a one-shot request and respond with the best answer possible using only the information and tools you have available.

--- OUTPUT FORMAT ---
Return your final answer as **valid Markdown** that Streamlit can directly render. For every news item you include, format it as a bullet in the following style:

- [Article Title](URL): one-sentence summary

Where *Article Title* is the actual headline or a short descriptive title and *(URL)* is a clickable link to the original article.  
Group related items under bolded section headers (e.g. **AI**, **Consumer Electronics**).  
At the end of your answer, add a "### Sources" heading followed by a numbered list of the links you cited so that users can easily verify them.

Ensure every link you cite corresponds to an item in your summary, and avoid including any links that were not referenced in the text above.

Here are the user's News Source Preferences. Only search these preferences:
{news_source_preferences}

Here are the user's Content Preferences. Only provide information that matches these preferences:
{content_preferences}

If your first tool call returns fewer than 5 unique, relevant results
   • Retry tavily_search with max_results=20 and search_depth="advanced".
   • If still insufficient, switch to tavily_map_site on the top domain,
     then tavily_extract_content for any promising URLs.
Stop iterating only when you have at least 5 articles that satisfy both
the News Source Preferences and the Content Preferences, or when three
consecutive tool calls fail to add new articles.
"""

    response = llm_with_tools.invoke(
        [{"role": "system", "content": system_prompt}, *state["messages"]]
    )

    update = {
        "messages": [response],
    }
    return Command(
        update=update,
        goto="tools",
    )


# Create the tool node with all available tools
tools = [tavily_crawl, tavily_map_site, tavily_search, tavily_extract_content]
tool_node = ToolNode(tools)


def feedback_node(state: CrawlState, store: BaseStore) -> Command[Literal["__end__"]]:
    """Feedback node introduces human in the loop to provide feedback on the results."""
    request = {
        "action_request": {
            "action": "Feedback: how did the agent do on the summary? Is there anything else you want to add or change?",
            "args": {},
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": False,
            "allow_accept": False,
        },
        "description": state["messages"][-1].content,
    }
    response = interrupt([request])[0]

    if response["type"] == "response":
        user_input = response["args"]
        state["messages"].append({"role": "user", "content": user_input})
        update_memory(
            store, ("news_feed_agent", "news_source_preferences"), state["messages"]
        )
        update_memory(
            store, ("news_feed_agent", "content_preferences"), state["messages"]
        )
        goto = END

    elif response["type"] == "ignore":
        goto = END

    else:
        raise ValueError(f"Invalid response: {response}")

    return Command(goto=goto)


# Create the workflow
def should_continue(state: CrawlState) -> str:
    """Determine whether to continue to tools or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# Keep the original test_graph for backward compatibility
def should_continue_messages(state: MessagesState) -> str:
    """Determine whether to continue to tools or end for MessagesState."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


overall_workflow = (
    StateGraph(MessagesState)
    .add_node("agent", lambda state, store: crawl_agent(state, store))
    .add_node("tools", tool_node)
    .add_node("feedback", feedback_node)
    .add_edge(START, "agent")
    .add_edge("tools", "agent")
    .add_conditional_edges(
        "agent",
        should_continue_messages,
        {
            "tools": "tools",
            END: "feedback",
        },
    )
    .add_edge("feedback", END)
)

news_agent = overall_workflow.compile()
