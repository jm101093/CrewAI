import os
import openai
from textwrap import dedent
from crewai import Agent, Task, Crew, Process, tools
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
#from langchain_community.tools import SleepTool 
from langchain_community.utilities import TextRequestsWrapper
from crewai_tools import WebsiteSearchTool
search_tool = DuckDuckGoSearchRun()
search_tool1 = DuckDuckGoSearchResults()
website_rag = WebsiteSearchTool()
from langchain_community.tools import DuckDuckGoSearchRun

from crewai_tools import tool

@tool('DuckDuckGoSearch')
def search(search_query: str):
    """Search the web for information on a given topic"""
    return DuckDuckGoSearchRun().run(search_query)

llm_lmstudio = ChatOpenAI(
    openai_api_key="null",
    openai_api_base="http://localhost:1234/v1",               
    model_name="llama"
)

class crew:
    def __init__(self, userInput):
        self.userInput = userInput

    def run(self):
        # Define your agents with roles and goals
        manager = Agent(
            role='Editor and chief',
            goal=f'Craft compelling content on {self.userInput} or similar terms that are relevant',
            backstory="""You are a renowned Editor and chief of a major media news outlet, known for your insightful and engaging articles using multiple sources.
            You manage a team that consists of your Content Strategist and your Senior Research Analyst. You will delagate research tasks to the Senior Research 
            Analyst to use search tools to bring back relavant information to the Content Strategist who will write the article. you will then take the article 
            written by the Content Strategist and review it and make sure all the given URL's are valid before giving having the Content Strategist make any edits you direct.""",
            verbose=True,
            tools=[website_rag],
            allow_delegation=True,
            llm=llm_lmstudio,
            max_iter=120,
            memory=True
        )
        researcher = Agent(
            role='Senior Research Analyst',
            goal=f'Search the internet for the latest trends and news on {self.userInput} or similar terms that are relevant',
            backstory="""You work at a leading Newsroom.
            Your expertise lies in identifying emerging trends and unique locations that regular tourists might miss.
            You have a knack for dissecting complex data and presenting actionable insights. You are also very good at 
            ignoring parts of articles that are advertisements or asking for social media engagements. 
            You can use your search_tools more than once.""",
            #Your Manager is the Editor and chief.
            verbose=True,
            allow_delegation=False,
            tools=[search, search_tool],
            llm=llm_lmstudio,
            max_iter=120,
            memory=True
        )
        writer = Agent(
            role='Content Strategist',
            goal=f'Craft compelling content on {self.userInput} or similar terms that are relevant into a 600 word or more blog post',
            backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles using multiple sources.
            You transform concepts into compelling narratives and .""",
            verbose=True,
            allow_delegation=True,
            llm=llm_lmstudio,
            max_iter=120,
            memory=True
        )

        # Create tasks for your agents
        mangementTask= Task(
            description="""You are a renowned Editor and chief of a major media news outlet, known for your insightful and engaging articles using multiple sources.
            You manage a team that consists of your Content Strategist and your Senior Research Analyst. You will delagate research tasks to the Senior Research 
            Analyst to use search tools to bring back relavant information to the Content Strategist who will write the article. you will then take the article 
            written by the Content Strategist and review it to make sure it provides valid working links to the locations websites or yelp pages.
            Do not use google to search for results before giving having the Content Strategist make any edits you direct""",
            agent=manager,
            expected_output=f"""A blog post that consist of information related to {self.userInput}""",
            async_execution=True
            
        )
        task1 = Task(
            description=f"""Conduct a comprehensive analysis of the latest {self.userInput} or similar terms that are relevant. 
            Use different search queries each time you call the search tool. Your final answer MUST be a full analysis report. 
            Use multiple sources for your information and Ignore the first 2 suggested links from search results.""",
            agent=researcher,
            expected_output=f"""Search results and content that consist of information related to the given request"""
        )

        task2 = Task(
            description=f"""Develop an engaging blog post that highlights {self.userInput} or similar terms that are relevant. 
            Your post should be informative yet accessible, catering to any audience. Make it sound cool, avoid complex words so it doesn't sound like AI. Also avoid sentences that have the term "Vote Now" and "shares"
            Your final answer MUST be the full blog post of around 1000 words or more. Also provide links to the locations websites or yelp pages. 
            You can delegate this last task of looking up those links to the Senior Research Analyst. """,
            #Your Manager is the Editor and chief send your final answer to him.
            agent=writer,
            expected_output=f"""A blog post that highlights {self.userInput} That is neetly formatted and is at least 600 words or more, with refrences, links, and any contact information relevant."""
        )
        task3 = Task(
            description=f"""Take any recommendations from the Editor and chief on your origional article and make revisions to your article. If you need to looking up any additional information 
            have the Senior Research Analyst will use the search_tools to find it for you before giving your final answer That should be in a blog post format of at least 1000 words or more. 
            Put all of the individual locations in a list format below the blog post with the locations and ike to any websites or URL's. 
            Also provide links to the locations websites, yelp pages or review sites as well as locations and contact information. Make it sound cool, avoid complex words so it doesn't sound like AI.""",
            agent=writer,
            expected_output=f"""A blog post that highlights {self.userInput} """
        )

        # Instantiate your crew with a sequential process
        crew = Crew(
            agents=[researcher, writer],
            tasks=[task1, task2],
            verbose=2, # You can set it to 1 or 2 to different logging levels
            #manager_llm=llm_lmstudio,#(temperature=0, model="llama"),
            process=Process.sequential
            #max_rpm=15
        )
        # Get your crew to work!
        return crew.kickoff()



if __name__ == "__main__":
    print("## Welcome to Newsletter Writer")
    print('-------------------------------')
    # Get user input of what to research like (What are the best vegan restaurants in san diego in 2024?, What is the solution to the Israeli and Palestinian conflict?)
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
