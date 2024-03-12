import os
import openai
from textwrap import dedent
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
#from langchain_community.tools import SleepTool 
from langchain_community.utilities import TextRequestsWrapper
from crewai_tools import WebsiteSearchTool
search_tool = DuckDuckGoSearchRun()
search_tool1 = DuckDuckGoSearchRun()
website_rag = WebsiteSearchTool()


llm_lmstudio = ChatOpenAI(
    openai_api_key="null",
    openai_api_base="http://localhost:1234/v1",                
    model_name="mistral"
)

class crew:
    def __init__(self, userInput):
        self.userInput = userInput

    def run(self):
        # Define your agents with roles and goals
        researcher = Agent(
            role='Senior Research Analyst',
            goal=f'Uncover the latest trends and news in {self.userInput} or similar terms that are relevant',
            backstory="""You work at a leading Newsroom.
            Your expertise lies in identifying emerging trends.
            You have a knack for dissecting complex data and presenting actionable insights. You are also very good at 
            ignoring parts of articles that are advertisements or asking for social media engagements. 
            You can use your search_tools more than once""",
            verbose=True,
            allow_delegation=True,
            tools=[search_tool, search_tool1, website_rag],
            llm=llm_lmstudio,
            max_iter=40,
            memory=True
        )
        writer = Agent(
            role='Content Strategist',
            goal=f'Craft compelling content on {self.userInput} or similar terms that are relevant',
            backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles using multiple sources.
            You transform concepts into compelling narratives.""",
            verbose=True,
            allow_delegation=True,
            llm=llm_lmstudio,
            max_iter=50,
            memory=True
        )

        # Create tasks for your agents
        task1 = Task(
            description=f"""Conduct a comprehensive analysis of the latest {self.userInput} or similar terms that are relevant.
            Your final answer MUST be a full analysis report. Use multiple sources for your information and Ignore the first 2 suggested links from search results.
            Take your time and be sure not to hit the rate limit on the search_tool. If you do pause and wait a second before continuing your work.""",
            agent=researcher,
            expected_output=f"""Search results that consist of information related to {self.userInput}"""
        )

        task2 = Task(
            description=f"""Don't use any tools but using the insights provided, develop an engaging blog
            post that highlights the {self.userInput} or similar terms that are relevant. 
            Your post should be informative yet accessible, catering to any audience.
            Make it sound cool, avoid complex words so it doesn't sound like AI. Also avoid sentences that have the term "Vote Now" and "shares"
            Your final answer MUST be the full blog post of around 600 words or more do not mention this in your final answer. Also provide links to the locations websites or yelp pages. 
            You can delegate this last task of looking up those links to the Senior Research Analyst that will use the search_tools""",
            agent=writer,
            expected_output=f"""A blog post that highlights {self.userInput} """
        )

        # Instantiate your crew with a sequential process
        crew = Crew(
            agents=[researcher, writer],
            tasks=[task1, task2],
            verbose=2, # You can set it to 1 or 2 to different logging levels
            process=Process.hierarchical,
            manager_llm=llm_lmstudio
        )
        # Get your crew to work!
        return crew.kickoff()



if __name__ == "__main__":
    print("## Welcome to Newsletter Writer")
    print('-------------------------------')
    # Get user input of what to research like (best vegan restaurants in san diego in 2024)
    userInput = input(
        dedent("""
            What is the topic you want to look up?
        """))

    crewresult = crew(userInput)
    result = crewresult.run()
    print("\n\n########################")
    print("## Here is the Result")
    print("########################\n")
    print(result)
